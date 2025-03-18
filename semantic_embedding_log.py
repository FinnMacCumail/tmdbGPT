import os
import logging
import json
import chromadb

def setup_file_logger(log_file: str) -> logging.Logger:
    """
    Configures and returns a logger that writes INFO-level logs to the specified file.
    """
    logger = logging.getLogger("ChromaLogger")
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def log_chroma_collection_contents(collection, logger: logging.Logger):
    """
    Retrieves all records from the given ChromaDB collection and logs them.
    """
    try:
        # Retrieve all stored records from the collection
        results = collection.get()
        # Log the results as a pretty-printed JSON string
        logger.info("Chroma Collection Contents:\n%s", json.dumps(results, indent=2))
    except Exception as e:
        logger.error("Error retrieving collection contents: %s", str(e))

# Initialize ChromaDB client and access the collection used by semantic_embed.py
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

# Ensure the logs folder exists
LOGS_FOLDER = "logs"
os.makedirs(LOGS_FOLDER, exist_ok=True)

# Define the output log file path
OUTPUT_LOG_FILE = os.path.join(LOGS_FOLDER, "chroma_embeddings.log")

# Setup the logger
logger = setup_file_logger(OUTPUT_LOG_FILE)

# Log the contents of the ChromaDB collection
log_chroma_collection_contents(collection, logger)
