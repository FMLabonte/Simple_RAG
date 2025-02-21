from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.postprocessor import SimilarityPostprocessor
import chromadb
import io
from openai import OpenAI
import os
from llama_index.core import SimpleDirectoryReader
import tempfile
import os
import shutil
import re
from typing import Union, List


##TODO add different splitting options
##TODO try with classes for cleaner code # works but not as pretty for UI use
##TODO(optional) add a two step function that first generates how a reply migh look to search with that

app = FastAPI()

# Store client instances for different users
db = chromadb.PersistentClient(path="./chroma_db")
user_collections = {}



@app.post("/ingest")
async def ingest_documents(
    db_name: str,
    recursive: bool = False,
    num_workers: int = 4,
    files: List[UploadFile] = File(...)
):
    try:
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to temporary directory
            saved_files = []
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                saved_files.append(file_path)

            # Use SimpleDirectoryReader to load documents
            reader = SimpleDirectoryReader(
                input_files=saved_files,  # Use specific files instead of directory
                recursive=recursive,
            )
            documents = reader.load_data(num_workers=num_workers)

            # Create or get user's collection
            collection_name = db_name
            chroma_collection = db.get_or_create_collection(collection_name)
            
            # Create vector store and storage context
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index with loaded documents
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            
            # Store the index for later use
            user_collections[db_name] = index
            
            return {
                "message": f"Successfully ingested {len(documents)} documents for user {db_name}",
                "filenames": [file.filename for file in files],
                "document_count": len(documents)
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query_openai")
async def query_openai(query: str):
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create completion using GPT-4 or GPT-3.5-turbo
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return {
            "answer": response.choices[0].message.content,
            "model": response.model,
            "query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def extract_node_info(node_with_score, similarity_cutoff=0.7):
    """Extract node info only if score is above cutoff"""
    if node_with_score.score >= similarity_cutoff:
        return {
            "text": node_with_score.node.text,
            "score": node_with_score.score,
            "metadata": node_with_score.node.metadata
        }
    return None

def extract_all_nodes_info(nodes_list, similarity_cutoff=0.7):
    """Filter and extract all nodes info above cutoff"""
    results = [extract_node_info(node, similarity_cutoff) for node in nodes_list]
    return [r for r in results if r is not None]

def filter_node_for_keyword(filtered_nodes: list, keywords: str, partial_match: bool = False) -> list:
    """
    Filter nodes using regex pattern matching for comma-separated keywords.
    
    Args:
        filtered_nodes: List of nodes to filter
        keywords: Comma-separated string of keywords (e.g., "keyword1,keyword2,keyword3")
        partial_match: If True, matches keywords anywhere in text
    
    Returns:
        List of matched nodes
    """
    if not keywords:
        return filtered_nodes
        
    # Split the comma-separated string into list and clean whitespace
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    # Create patterns for all keywords
    patterns = []
    for keyword in keyword_list:
        # Normalize dashes in keyword
        normalized_keyword = keyword.replace('-', '[–-]')
        
        if partial_match:
            # Match keyword anywhere, handling quotes and tags
            pattern = re.compile(f'.*{normalized_keyword}.*?["/]?.*', re.IGNORECASE)
        else:
            # Match exact keyword with word boundaries
            pattern = re.compile(f'\\b{normalized_keyword}\\b', re.IGNORECASE)
        patterns.append(pattern)
    
    # Filter nodes that match any of the patterns
    return [
        node for node in filtered_nodes 
        if any(pattern.search(node["text"]) for pattern in patterns)
    ]

@app.post("/retrieve_sources")
async def retrieve_sources(
    query: str,
    keywords: str = None, # Comma-separated keywords
    similarity_cutoff: float = 0.7,
    top_k: int = 5,
    database_name: str = "sampled_papers",
    partial_match: bool = False
):
    try:
        # initialize client
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_collection(database_name)
        
        # setup vector store and index
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context
        )

        # use retriever instead of query engine
        retriever = index.as_retriever(
            similarity_top_k=top_k*4,  # retrieve more to filter later necessary for keyword
        )

        # get nodes
        nodes = retriever.retrieve(query)

        # filter and process nodes
        filtered_nodes = extract_all_nodes_info(nodes, similarity_cutoff)
        
        if keywords is not None:
            filtered_nodes = filter_node_for_keyword(filtered_nodes, keywords, partial_match)

        # limit to top_k after filtering
        filtered_nodes = filtered_nodes[:top_k]



        return {
            "total_nodes_retrieved": len(nodes),
            "nodes_after_filtering": len(filtered_nodes),
            "cutoff_score": similarity_cutoff,
            "sources": filtered_nodes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query_with_context")
async def query_with_context(
    query: str,
    keywords: str = None, # Comma-separated keywords
    similarity_cutoff: float = 0.7,
    top_k: int = 5,
    database_name: str = "sampled_papers",
    partial_match: bool = False
):
    try:
        # Get sources using existing endpoint
        sources_response = await retrieve_sources(query =query, keywords=keywords,similarity_cutoff= similarity_cutoff, top_k=top_k, database_name=database_name, partial_match=partial_match)
        
        # Prepare context from retrieved sources
        sources_context = "\n\n".join([
            f"Source {i+1}:\n{source['text']}"
            for i, source in enumerate(sources_response["sources"])
        ])

        # Create prompt for OpenAI
        prompt = f"""Based on the following sources, answer this question: {query}

Sources:
{sources_context}

Please provide a comprehensive answer using only the information from these sources. 
If the sources don't contain relevant information, please state that."""

        # Use existing OpenAI query endpoint
        ai_response = await query_openai(prompt)

        # Return combined response
        return {
            "answer": ai_response["answer"],
            "model": ai_response["model"],
            "sources_info": sources_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/list_databases")
async def list_databases():
    """
    Retrieve all database (collection) names and their document counts from ChromaDB.
    
    Returns:
        dict: Dictionary containing list of all database names and their document counts
    """
    try:
        # Get all collection names
        database_names = db.list_collections()
        
        # Get collection info with document counts
        database_info = []
        for name in database_names:
            collection = db.get_collection(name)
            database_info.append({
                "name": name,
                "document_count": collection.count()
            })
        
        return {
            "total_collections": len(database_names),
            "collections": database_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/remove_database/{database_name}")
async def remove_database(database_name: str):
    """
    Remove a specific database (collection) from ChromaDB.
    
    Args:
        database_name: Name of the database/collection to remove
    
    Returns:
        dict: Status message indicating success or failure
    """
    try:
        # Check if collection exists
        collections = db.list_collections()
        if database_name not in [col for col in collections]:
            raise HTTPException(
                status_code=404,
                detail=f"Database '{database_name}' not found"
            )
        
        # Delete the collection
        db.delete_collection(database_name)
        
        return {
            "status": "success",
            "message": f"Database '{database_name}' successfully removed",
            "remaining_collections": len(db.list_collections())
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)