from chromadb import PersistentClient

# Path to ChromaDB storage
CHROMA_DB_PATH = "/home/ola/ollamadev/tmdbRest/embeddings/chroma_db"

# Connect to ChromaDB
chroma_client = PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

# Print stored embeddings
print(f"🔍 Total stored embeddings: {collection.count()}\n")

results = collection.get()
for i, (id, metadata) in enumerate(zip(results["ids"], results["metadatas"])):
    print(f"📌 Entry {i+1}:")
    print(f"✅ ID: {id}")
    print(f"✅ Metadata: {metadata}")
    print("=" * 50)
