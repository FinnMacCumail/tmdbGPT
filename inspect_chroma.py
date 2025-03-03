import chromadb
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("inspect_chroma")

# Use the correct database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#CHROMA_DB_PATH = os.path.join(BASE_DIR, "..", "embeddings", "chroma_db")
CHROMA_DB_PATH = "embeddings/chroma_db"

logger.info(f"📂 Using ChromaDB Path: {CHROMA_DB_PATH}")

# Connect to ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="tmdb_queries")  # Ensure correct collection name
except Exception as e:
    logger.error(f"❌ Error initializing ChromaDB: {e}")
    exit(1)

# Inspect stored queries
def inspect_chromadb():
    logger.info("🔍 Inspecting stored TMDB queries in ChromaDB...")

    try:
        stored_data = collection.get()
        stored_ids = stored_data.get("ids", []) if stored_data else []
        stored_docs = stored_data.get("documents", []) if stored_data else []

        logger.info(f"📂 Found {len(stored_ids)} stored queries in ChromaDB.")

        if len(stored_ids) == 0:
            logger.error("❌ No stored queries found in ChromaDB!")
            return

        # Debug: Print first few stored queries
        first_queries = stored_docs[:5]  # Use "documents" instead of "metadatas"
        if first_queries:
            logger.info("📝 Sample stored queries in ChromaDB:")
            for i, query in enumerate(first_queries):
                logger.info(f"   {i+1}. Query: {query}")
        else:
            logger.warning("⚠️ No documents found in stored queries.")

    except Exception as e:
        logger.error(f"❌ Error inspecting ChromaDB: {e}")

if __name__ == "__main__":
    inspect_chromadb()
