import os
import json
import chromadb

# Ensure logs directory exists
LOGS_DIR = "./logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# File path for the log output
LOG_FILE_PATH = os.path.join(LOGS_DIR, "semantic_embeddings.log")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./semantic_chroma_db")
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

def list_semantic_embeddings():
    """Fetch and log all stored semantic embeddings from ChromaDB while handling missing values."""
    try:
        # ✅ Fetch only embeddings and metadata
        all_data = collection.get(include=["embeddings", "metadatas"])

        if not all_data or "metadatas" not in all_data or not all_data["metadatas"]:
            print("⚠️ No embeddings found in ChromaDB!")
            return
        
        log_entries = []

        for idx, metadata in enumerate(all_data["metadatas"]):
            # ✅ Extract ID from metadata (ChromaDB stores IDs inside metadata)
            api_id = metadata.get("path", f"unknown_{idx}")

            # ✅ Handle missing embeddings gracefully
            embedding = all_data.get("embeddings", [None])[idx]  
            
            entry = {
                "id": api_id,
                "embedding": embedding if embedding is not None else "❌ Missing Embedding",
                "metadata": metadata if metadata is not None else "❌ Missing Metadata"
            }
            log_entries.append(entry)

        # ✅ Write logs to file
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as log_file:
            json.dump(log_entries, log_file, indent=2)

        print(f"✅ Logged {len(log_entries)} embeddings to {LOG_FILE_PATH}")

    except Exception as e:
        print(f"❌ Error fetching embeddings: {str(e)}")

if __name__ == "__main__":
    list_semantic_embeddings()
