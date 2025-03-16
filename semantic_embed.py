import json
import chromadb
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables (if needed)
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path, override=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./semantic_chroma_db")
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load TMDB API schema
TMDB_SCHEMA_PATH = "./data/tmdb.json"
with open(TMDB_SCHEMA_PATH, "r", encoding="utf-8") as f:
    tmdb_schema = json.load(f)

api_endpoints = tmdb_schema.get("paths", {})


def infer_api_context(path, description, query_params, path_placeholders):
    """
    Dynamically infers API usage context based on path patterns and extracted parameters.
    Ensures search-first logic for missing ID dependencies.
    """

    # ✅ Identify if an endpoint requires an ID
    requires_id = any(placeholder != "None" for placeholder in path_placeholders)

    # ✅ Determine if it's a "search-first" API (uses query params instead of an ID)
    is_query_based = len(query_params) > 0 and not requires_id

    if "search" in path:
        return "used for searching"
    if requires_id and "movie_credits" in path:
        return "retrieves movies directed by a person"
    if requires_id and "credits" in path:
        return "retrieves cast and crew of a movie"
    if "/tv/" in path:
        return "retrieves TV show information"
    if "/network/" in path:
        return "retrieves network-related data"
    if "/company/" in path:
        return "retrieves company-related data"
    if "/collection/" in path:
        return "retrieves movie collections"

    # ✅ Prioritize APIs that can be executed immediately (search/query-based)
    if is_query_based:
        return "query-based API, no ID required"
    
    return "ID-based API, requires entity resolution"


def extract_parameters(endpoint_data):
    """
    Extracts query parameters vs. path placeholders dynamically.
    Ensures proper classification for ranking and execution.
    """
    parameters = endpoint_data.get("parameters", [])
    query_params = []
    path_placeholders = []

    for param in parameters:
        param_name = param.get("name")
        if param.get("in") == "query":
            query_params.append(param_name)
        elif param.get("in") == "path":
            path_placeholders.append(param_name)

    return query_params, path_placeholders


def store_api_embeddings():
    """
    Processes all API endpoints, generates embeddings, and stores them in ChromaDB.
    Dynamically flags ID-dependent APIs vs. query-based APIs.
    """

    all_api_data = []

    for path, methods in api_endpoints.items():
        for method, data in methods.items():
            description = data.get("summary", "No description available")
            query_params, path_placeholders = extract_parameters(data)
            context = infer_api_context(path, description, query_params, path_placeholders)

            # ✅ Generate full context description for embedding
            full_text = (
                f"{description} | Context: {context} | Path: {path} | Method: {method} | "
                f"Query Params: {', '.join(query_params) if query_params else 'None'} | "
                f"Path Placeholders: {', '.join(path_placeholders) if path_placeholders else 'None'}"
            )

            # ✅ Generate embedding vector
            embedding = model.encode(full_text, convert_to_numpy=True).tolist()

            # ✅ Store in ChromaDB with enhanced metadata
            collection.add(
                ids=[path],  # Use path as unique identifier
                embeddings=[embedding],
                metadatas=[
                    {
                        "path": path,
                        "method": method,
                        "description": description,
                        "context": context,  # ✅ Now dynamically generated
                        "query_params": ", ".join(query_params) if query_params else "None",
                        "path_placeholders": ", ".join(path_placeholders) if path_placeholders else "None"
                    }
                ],
            )

            all_api_data.append({
                "path": path,
                "method": method,
                "description": description,
                "context": context,
                "query_params": query_params,
                "path_placeholders": path_placeholders
            })

    print(f"✅ Successfully stored {len(all_api_data)} API endpoints in ChromaDB!")


if __name__ == "__main__":
    store_api_embeddings()
