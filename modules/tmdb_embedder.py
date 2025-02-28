import json
import chromadb
from sentence_transformers import SentenceTransformer
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "tmdb.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")

# Initialize ChromaDB client (persistent storage)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

def normalize_path(path):
    """Keep TMDB path parameters in curly braces."""
    return path

def extract_parameters(param_list):
    """Convert API parameter list into a structured dictionary."""
    parameters = {}
    for item in param_list:
        if isinstance(item, dict):
            name = item.get("name")
            desc = item.get("description", "No description available")
            if name:
                parameters[name] = desc
    return parameters

def generate_dynamic_query(path, method, parameters):
    """
    Dynamically generate a query description based only on endpoint structure and available parameters.
    """
    # Extract words from the API path (ignoring placeholders like {movie_id})
    path_tokens = [token for token in path.strip("/").split("/") if not token.startswith("{")]
    formatted_path = " ".join(path_tokens).replace("_", " ")

    # Convert parameters into a meaningful sentence
    param_descriptions = [f"{name} ({desc})" for name, desc in parameters.items() if desc != "No description available"]
    param_text = ", ".join(param_descriptions) if param_descriptions else "relevant filters"

    # Use schema-derived names instead of predefined labels
    query_text = f"Retrieve {formatted_path} using {param_text}."

    return query_text

def load_tmdb_schema():
    """Load TMDB OpenAPI schema and extract query-based information."""
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        api_schema = json.load(file)

    query_mappings = []

    for path, methods in api_schema.get("paths", {}).items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                print(f"⚠️ Warning: Skipping malformed entry at {path} - {details}")
                continue

            parameters = {}
            path_params = []

            if "parameters" in details and isinstance(details["parameters"], list):
                for param in details["parameters"]:
                    if isinstance(param, dict) and "name" in param:
                        param_name = param["name"]
                        param_type = param["schema"]["type"] if "schema" in param else "string"

                        if param.get("in") == "path":
                            path_params.append(param_name)  # ✅ Store path params
                        else:
                            parameters[param_name] = param.get("description", f"No description available for {param_name}")

            # ✅ Replace placeholders dynamically in path
            normalized_path = path
            for param in path_params:
                normalized_path = normalized_path.replace(f"{{{param}}}", f"<{param}>")  # Better readability

            # ✅ Ensure paths are stored properly (No skipping!)
            query_text = f"Retrieve `{normalized_path}` using `{method.upper()}`."

            query_mappings.append({
                "query": query_text,
                "solution": {
                    "endpoint": normalized_path,
                    "method": method.upper(),
                    "parameters": parameters
                }
            })

    return query_mappings


def embed_tmdb_queries():
    """Embed TMDB queries into ChromaDB."""
    print("🛠️ Debug: Checking existing collections...")
    
    existing_collections = chroma_client.list_collections()
    print("✅ Available collections:", existing_collections)

    print("🔍 Extracting TMDB API mappings from OpenAPI schema...")
    query_mappings = load_tmdb_schema()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🗑️ Checking existing embeddings...")
    existing_items = collection.get()
    if existing_items and "ids" in existing_items:
        stored_ids = existing_items["ids"]
        if stored_ids:
            print(f"🗑️ Deleting {len(stored_ids)} old embeddings...")
            collection.delete(ids=stored_ids)
        else:
            print("✅ No old embeddings found.")
    else:
        print("✅ No old embeddings found.")

    queries = []
    metadata = []

    for i, entry in enumerate(query_mappings):
        query_text = entry.get("query", "").strip()
        solution = entry.get("solution", {})

        if query_text and solution:
            queries.append(query_text)
            metadata.append({"solution": json.dumps(solution)})

    print("🔄 Generating new embeddings...")
    query_vectors = model.encode(queries).tolist()

    collection.add(
        ids=[str(i) for i in range(len(queries))],
        embeddings=query_vectors,
        metadatas=metadata
    )

    print(f"✅ {len(queries)} TMDB queries embedded successfully in ChromaDB!")

if __name__ == "__main__":
    embed_tmdb_queries()
