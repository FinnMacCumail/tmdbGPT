import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Path configurations
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TMDB_SCHEMA_PATH = os.path.join(BASE_DIR, "data", "tmdb.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

def extract_parameters(param_list):
    """Extracts parameters from a list, handling both dictionaries and lists properly."""
    combined = {}
    if isinstance(param_list, list):  # Ensure it's a list
        for param in param_list:
            if isinstance(param, dict):  # Check if each entry is a dictionary
                name = param.get("name")
                desc = param.get("description", "No description available")
                if name:
                    combined[name] = desc
    return combined

def load_tmdb_schema():
    """Load TMDB OpenAPI schema and extract query-based information."""
    with open(TMDB_SCHEMA_PATH, "r", encoding="utf-8") as file:
        api_schema = json.load(file)

    query_mappings = []
    
    for path, methods in api_schema.get("paths", {}).items():
        for method, details in methods.items():
            if isinstance(details, list):  
                details = {"parameters": extract_parameters(details)}  
            elif not isinstance(details, dict):
                print(f"⚠️ Warning: Skipping malformed entry at {path} - {details}")
                continue

            # Fix the `parameters` extraction issue
            parameters = extract_parameters(details.get("parameters", [])) if "parameters" in details else {}

            normalized_path = path  # Ensure path is correctly formatted

            query_text = details.get("summary", f"What is {normalized_path}?")

            query_mappings.append({
                "query": query_text,
                "solution": {
                    "endpoint": normalized_path,
                    "method": method.upper(),
                    "parameters": parameters
                }
            })

    return query_mappings

def normalize_path(path):
    """Replace path parameters with a placeholder name."""
    return path.replace("{", "").replace("}", "").replace("/", " ")

def create_embedding_text(endpoint, data):
    """Generate a descriptive text representation for embedding."""
    param_text = ", ".join(f"{k}: {v}" for k, v in data["parameters"].items()) or "No parameters"
    return (
        f"API Endpoint: {endpoint}. Description: {data['description']}. Parameters: {param_text}."
    )

def embed_tmdb_queries():
    """Embed TMDB queries into ChromaDB."""
    print("🔍 Extracting TMDB API mappings from OpenAPI schema...")

    # Load the TMDB API schema and extract relevant data
    query_mappings = load_tmdb_schema()  # This returns a LIST

    print("🗑 Checking existing embeddings...")

    # Delete old embeddings
    existing_items = collection.get()
    if existing_items and "ids" in existing_items:
        stored_ids = existing_items["ids"]
        if stored_ids:
            print(f"🗑 Deleting {len(stored_ids)} old embeddings...")
            collection.delete(ids=stored_ids)
        else:
            print("✅ No old embeddings found. Proceeding with new embeddings.")
    else:
        print("✅ No old embeddings found. Proceeding with new embeddings.")

    queries = []
    metadata = []
    embeddings = []

    # Corrected iteration over the list
    for i, entry in enumerate(query_mappings):  
        query_text = entry.get("query", "").strip()
        solution = entry.get("solution", {})

        if query_text and solution:
            queries.append(query_text)
            metadata.append({"solution": json.dumps(solution)})

    # Convert queries into embeddings
    print("🔄 Generating new embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vectors = model.encode(queries).tolist()

    # Store in ChromaDB
    collection.add(
        ids=[str(i) for i in range(len(queries))],  
        embeddings=query_vectors,
        metadatas=metadata
    )

    print(f"✅ {len(queries)} TMDB queries embedded successfully in ChromaDB!")


if __name__ == "__main__":
    embed_tmdb_queries()
