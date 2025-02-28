from chromadb import PersistentClient

# Ensure this path matches your TMDB ChromaDB path
CHROMA_DB_PATH = "/home/ola/ollamadev/tmdbRest/embeddings/chroma_db"

# Connect to ChromaDB
chroma_client = PersistentClient(path=CHROMA_DB_PATH)

# Get or create the collection
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

# Check how many embeddings exist
print(f"Total stored embeddings: {collection.count()}")

results = collection.query(query_texts=["Christopher Nolan"], n_results=5)
print(results)
