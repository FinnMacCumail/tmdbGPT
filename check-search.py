import chromadb
import os
import json

# Set ChromaDB path
CHROMA_DB_PATH = "embeddings/chroma_db"

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection("tmdb_queries")

# Check stored query for "/search/person"
search_query = "/search/person"
stored_query = collection.query(query_texts=[search_query], n_results=1)

print(f"🔍 Stored ChromaDB Query for {search_query}: {stored_query}")
