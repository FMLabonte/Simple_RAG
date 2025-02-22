# Simple_RAG
A simple FastAPI wrapper around Llama-index and OpenAI's API.

## Description
This project implements a simple RAG (Retrieval-Augmented Generation) pipeline using Llama Index and OpenAI's ChatGPT to answer questions based on provided documents. Llama Index handles the vectorization, and the vector databases are stored using ChromaDB.

I have written multiple endpoints:

## Endpoints

### Ingest
Accepts PDFs and other text documents and stores them under a named database, which can then be accessed in other endpoints.

### Query_OpenAI
A direct wrapper for the OpenAI API if you want to ask questions without using the RAG pipeline.

### Retrieve_Sources
Retrieves sources from one of the databases created with the Ingest endpoint, without directly sending the results to ChatGPT. It implements filters for the number of sources to consider, the minimum similarity score, and a keyword function to allow only sources that contain a specific keyword.

### Query_With_Context
Combines the functionality of the Query_OpenAI and Retrieve_Sources endpoints. It takes the same arguments as the Retrieve_Sources endpoint.

### List_Databases
Provides a list of all existing databases.

### Remove_Database
Allows you to delete a database by name.

## how to run
To run the docker container clone the git reposetory.
create an .env file with your open AI api key if you have set that one as a system variable do the following

```
cd Simple_RAG # go in to the RAG folder 
echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> .env #copy over the API key to the .env file
```
if you dont have the key set as a system variable you can also manually create an .env file in the Simple_RAG folder next to the dockerfile.
it must contain the following info
```
OPENAI_API_KEY=<your_API_key>
```
Then you can use docker to build the container and automatically start the api.
```
cd Simple_RAG # go in to the RAG folder 
docker compose up
```
Lastly visit http://localhost:8000/docs to use the API

## Future Ideas
To improve performance, more advanced or different ways of indexing could be implemented. For example, the Ingest endpoint could allow setting the chunk size or chunking by paragraph. For larger collections, summarizing document overviews and then searching further if a hit is found in the summarization could be beneficial. Additionally, a technique where the LLM generates a response first and then searches that against the database could be explored. To find optimal patterns, it would be necessary to construct a test case and then optimize performance based on it.
Another nice addition would be the use of Local LLMs through for example VLLM or llama.cpp by making use of their openai api compatible endpoints. 

