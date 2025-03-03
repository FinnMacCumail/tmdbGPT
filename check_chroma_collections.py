import os
import chromadb

# Define the ChromaDB path
CHROMA_DB_PATH = os.path.expanduser("~/embeddings/chroma_db")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# List all collections in the database
collections = chroma_client.list_collections()

# Print available collections
print("Available Collections in ChromaDB:")
for col in collections:
    print(f"- {col.name}")

# Check if 'tmdb_queries' exists
if any(col.name == "tmdb_queries" for col in collections):
    collection = chroma_client.get_collection("tmdb_queries")
    print(f"\n✅ Collection 'tmdb_queries' found. It contains {collection.count()} items.")
else:
    print("\n❌ Collection 'tmdb_queries' does not exist.")
