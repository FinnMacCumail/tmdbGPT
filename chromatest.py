import chromadb

# Use the expected database path
CHROMA_DB_PATH = "embeddings/chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

# Retrieve stored queries
data = collection.get()
print(f"Stored Queries: {len(data.get('ids', []))}")  # Should match the number of embedded queries
print(f"First 5 Queries: {data.get('documents', [])[:5]}")
