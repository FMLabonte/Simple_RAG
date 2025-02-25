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

## How to Run
To run the Docker container, clone the Git repository. 
```
git clone https://github.com/FMLabonte/Simple_RAG.git
```
Create an `.env` file with your OpenAI API key. If you have set that as a system variable, follow these steps:

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

Another nice addition would be the use of Local LLMs through, for example, VLLM or llama.cpp by making use of their OpenAI API-compatible endpoints.

## To test the RAG pipeline 
you can download the free pf rule book of Lancer a table top RPG which is rather unknown here: https://massif-press.itch.io/corebook-pdf-free and ingest the PDF in to the API.
you can ask for example "what is the black witch mech", "what mechs are produced by horus" and compare those outputs with the Query OpenAI endpoint which hallucinates or provides maretebly different answers.

you can of course test with yourn own documents. Use the Retrive sources endpoint to get an idea of if you need to shift some of the filters arround to retrive documents that seme relevant.
Compare the answers of the Query with context endpoint with the query openAI endpoint. 

### Further Remarks
The current script does not use classes due to some initial bugs with the UI elements. I decided, for time reasons and the rather small scale, to just use a direct implementation. For bigger projects and with a bit more time to figure out why the UI didn't like the Pydantic classes, that would be the chosen approach.

