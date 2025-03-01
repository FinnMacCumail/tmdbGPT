import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
import re

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "tmdb.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")

# Initialize ChromaDB client (persistent storage)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Create or get a collection for TMDB queries
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

def normalize_path(path):
    """Keep TMDB path parameters in curly braces instead of replacing with 'id'."""
    return path  # Don't replace parameters!

def extract_parameters_from_list(param_list):
    """Given a list of parameter definitions, combine them into a dictionary."""
    combined = {}
    for item in param_list:
        if isinstance(item, dict):
            name = item.get("name")
            desc = item.get("description", "No description available")
            if name:
                combined[name] = desc
    return combined

def generate_fallback_query(path):
    """Generate a human-readable fallback query for endpoints that lack a summary."""
    norm_path = normalize_path(path).replace("_", " ")
    if "/movie/id/credits" in norm_path:
        return "Retrieve the cast and crew for a movie"
    elif "/movie/id/recommendations" in norm_path:
        return "Get movie recommendations similar to a given movie"
    elif "/movie/id/similar" in norm_path:
        return "Find movies similar to a specific movie"
    elif "/movie/id/reviews" in norm_path:
        return "Get user reviews for a movie"
    elif "/movie/id/release_dates" in norm_path:
        return "Retrieve the release dates for a movie"
    elif "/tv/id/credits" in norm_path:
        return "Retrieve the cast and crew for a TV show"
    elif "/tv/id/similar" in norm_path:
        return "Find TV shows similar to a given show"
    elif "/tv/id/recommendations" in norm_path:
        return "Get TV show recommendations"
    elif "/person/id/movie_credits" in norm_path:
        return "Get all movie credits for a person"
    elif "/person/id/tv_credits" in norm_path:
        return "Get all TV credits for a person"
    elif "/person/id/images" in norm_path:
        return "Retrieve images for a person"
    elif "/collection/id" in norm_path:
        return "Retrieve information about a movie collection"
    elif "/company/id" in norm_path:
        return "Retrieve details about a production company"
    elif "/network/id" in norm_path:
        return "Retrieve details about a TV network"
    elif "trending" in norm_path:
        return "Retrieve trending media"
    return f"What is {norm_path}?"

def load_tmdb_schema():
    """Load TMDB OpenAPI schema and extract query-based information."""
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        api_schema = json.load(file)

    query_mappings = []
    
    # Iterate over each endpoint path in the schema
    for path, methods in api_schema.get("paths", {}).items():
        for method, details in methods.items():
            if isinstance(details, list):
                details = {"parameters": extract_parameters_from_list(details)}
            elif not isinstance(details, dict):
                print(f"⚠️ Warning: Skipping malformed entry at {path} - {details}")
                continue

            # Extract parameters if present
            parameters = {}
            if "parameters" in details:
                if isinstance(details["parameters"], list):
                    parameters = extract_parameters_from_list(details["parameters"])
                elif isinstance(details["parameters"], dict):
                    parameters = details["parameters"]
            
            # Normalize the endpoint path
            normalized_path = normalize_path(path)

            # Use the summary if available; otherwise, generate a fallback description
            query_text = details.get("summary", generate_fallback_query(path))

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

    # Load the TMDB API schema and extract relevant data
    query_mappings = load_tmdb_schema()

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # **Fix: Properly delete old embeddings**
    print("🗑️ Checking existing embeddings...")

    existing_items = collection.get()  # Get all stored IDs
    if existing_items and "ids" in existing_items:
        stored_ids = existing_items["ids"]
        if stored_ids:
            print(f"🗑️ Deleting {len(stored_ids)} old embeddings...")
            collection.delete(ids=stored_ids)  # Delete by IDs
        else:
            print("✅ No old embeddings found. Proceeding with new embeddings.")
    else:
        print("✅ No old embeddings found. Proceeding with new embeddings.")

    queries = []
    metadata = []
    embeddings = []

    for i, entry in enumerate(query_mappings):
        query_text = entry.get("query", "").strip()
        solution = entry.get("solution", {})

        if query_text and solution:
            queries.append(query_text)
            metadata.append({"solution": json.dumps(solution)})  # Store as JSON string

    # Convert queries into embeddings
    print("🔄 Generating new embeddings...")
    query_vectors = model.encode(queries).tolist()

    # Store in ChromaDB
    collection.add(
        ids=[str(i) for i in range(len(queries))],  # Ensure unique IDs
        embeddings=query_vectors,
        metadatas=metadata
    )

    print(f"✅ {len(queries)} TMDB queries embedded successfully in ChromaDB!")

if __name__ == "__main__":
    embed_tmdb_queries()
