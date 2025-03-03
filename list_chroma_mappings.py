import chromadb
import os
import json
import logging

# Ensure logs directory exists
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)  # ✅ Creates logs/ directory if missing

# Define log file path
LOG_FILE = os.path.join(LOGS_DIR, "chroma_mappings.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set ChromaDB path
CHROMA_DB_PATH = "embeddings/chroma_db"

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection("tmdb_queries")

# Fetch all stored mappings
all_queries = collection.get()  # Retrieve all data from ChromaDB

# Extract stored metadata
ids = all_queries.get("ids", [])
metadatas = all_queries.get("metadatas", [])

# Log header
logging.info("🔍 **ChromaDB Stored Mappings:**\n")

for idx, metadata in zip(ids, metadatas):
    solution = metadata.get("solution", "No solution found")
    try:
        solution_dict = json.loads(solution)  # Ensure it's formatted properly
    except json.JSONDecodeError:
        solution_dict = solution  # Keep as string if parsing fails

    log_entry = f"\n🆔 **ID:** {idx}\n📌 **Stored Mapping:**\n{json.dumps(solution_dict, indent=2)}\n{'=' * 50}\n"
    logging.info(log_entry)

print(f"✅ ChromaDB mappings logged to {LOG_FILE}")
