import json
import os
import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

# Define log directory
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "chroma_db_inspection.json")

def list_chroma_db_entries():
    """Retrieve stored entries from ChromaDB and save them to a log file."""
    print("🔍 Retrieving stored entries from ChromaDB...")
    
    # Fetch all stored data
    stored_data = collection.get()
    
    if not stored_data or "ids" not in stored_data:
        print("⚠️ No data found in ChromaDB collection.")
        return
    
    # Structure the data for logging
    extracted_entries = []
    for i, entry_id in enumerate(stored_data["ids"]):
        metadata = stored_data["metadatas"][i]
        extracted_entries.append({
            "id": entry_id,
            "description": stored_data["documents"][i],
            "metadata": metadata
        })
    
    # Write to log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_entries, f, indent=4, ensure_ascii=False)
    
    print(f"✅ ChromaDB data logged to {LOG_FILE}")

if __name__ == "__main__":
    list_chroma_db_entries()
