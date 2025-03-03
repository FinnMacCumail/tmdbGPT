import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
import openai
from dotenv import load_dotenv
from endMap import TMDB_API_MAP
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tmdb_embedder")

# Ensure the correct module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/tmdb.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "..", "embeddings", "chroma_db")  # Store ChromaDB above "modules"

# Initialize ChromaDB client (persistent storage)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI client
client = openai.Client(api_key=OPENAI_API_KEY)

def normalize_path(path):
    """Normalize TMDB path parameters to maintain consistency."""
    return path

def extract_parameters_from_list(param_list):
    """Extract parameters from the API schema for embedding purposes."""
    return {item["name"]: item.get("description", "No description") for item in param_list if isinstance(item, dict)}

def load_tmdb_schema():
    """Load TMDB OpenAPI schema and extract query-based information."""
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        api_schema = json.load(file)

    query_mappings = []
    
    for path, methods in api_schema.get("paths", {}).items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                logger.warning(f"⚠️ Skipping malformed entry at {path} - {details}")
                continue

            parameters = extract_parameters_from_list(details.get("parameters", []))
            normalized_path = normalize_path(path)
            description = details.get("summary", f"Query for {normalized_path}")

            # ✅ Generate queries dynamically using OpenAI function calling
            semantic_queries = generate_semantic_queries(normalized_path, description)

            # ✅ Debugging: Print generated queries to verify correctness
            logger.debug(f"Generated Queries for {normalized_path}: {semantic_queries}")

            if not semantic_queries:
                logger.error(f"🚨 ERROR: No valid queries generated for {normalized_path}")
                continue

            for query_text in semantic_queries:
                query_mappings.append({
                    "query": query_text,
                    "solution": {
                        "endpoint": normalized_path,
                        "method": method.upper(),
                        "parameters": parameters
                    }
                })

    return query_mappings

def generate_semantic_queries(endpoint_path, description):
    """
    Use OpenAI function calling to enforce correct response format for TMDB API query generation.
    """
    system_prompt = (
        "You are an AI that generates human-friendly queries for The Movie Database (TMDB) API. "
        "Your goal is to create diverse, user-friendly queries that match API endpoints "
        "without explicit rule-based mappings.\n"
        "Ensure queries are natural, relevant, and correctly formatted."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_queries",
                "description": "Generate 5 diverse queries for a given API endpoint.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "A list of five natural language queries for the API endpoint."
                        }
                    },
                    "required": ["queries"]
                }
            }
        }
    ]

    logger.info(f"\n🚀 Calling OpenAI to generate queries for: {endpoint_path}")

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate queries for: {endpoint_path} - {description}"}
        ],
        tools=tools,
        tool_choice="auto",
        max_tokens=200
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        logger.warning(f"⚠️ Warning: OpenAI returned no tool calls for {endpoint_path}.")
        return ["MISSING_QUERY"]  # ✅ Prevents `None` values

    tool_response = json.loads(tool_calls[0].function.arguments)
    queries = tool_response.get("queries", [])

    # ✅ Ensure all queries are valid strings
    queries = [q if isinstance(q, str) and q.strip() != "" else "MISSING_QUERY" for q in queries]

    logger.debug(f"📝 OpenAI Generated Queries for {endpoint_path}: {queries}")

    return queries

LOG_DIR = os.path.join(BASE_DIR, "../logs")  # Create a logs directory
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the directory exists
QUERY_MAPPING_LOG_FILE = os.path.join(LOG_DIR, "query_mappings.json")

def log_query_mappings(query_mappings):
    """Writes query mappings to a JSON file for inspection."""
    try:
        with open(QUERY_MAPPING_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(query_mappings, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Query mappings logged to {QUERY_MAPPING_LOG_FILE}")
    except Exception as e:
        logger.error(f"❌ Failed to log query mappings: {e}", exc_info=True)

def embed_tmdb_queries():
    """Embed TMDB queries into ChromaDB with debugging."""
    try:
        query_mappings = load_tmdb_schema()

        if not query_mappings or all(entry["query"] is None or entry["query"] == "" for entry in query_mappings):
            logger.error("🚨 ERROR: No valid queries found in query_mappings! Fix before embedding.")
            return

        # Log query mappings to file
        log_query_mappings(query_mappings)
        
        # Fetch existing stored queries from ChromaDB
        existing_data = collection.get()
        stored_ids = existing_data.get("ids", []) if existing_data else []

        logger.info(f"🔎 Before embedding: {len(stored_ids)} existing items in ChromaDB.")

        # Ensure old embeddings are only deleted if necessary
        if stored_ids:
            collection.delete(ids=stored_ids)
            logger.info(f"⚠️ Deleted {len(stored_ids)} old embeddings before inserting new ones.")

        queries = [entry["query"] if entry["query"] else "MISSING_QUERY" for entry in query_mappings]
        metadata = [{"solution": json.dumps(entry["solution"])} for entry in query_mappings]

        # ✅ Debugging: Print the first 10 queries before storing
        logger.info("\n📝 Storing the following queries in ChromaDB:")
        for i, q in enumerate(queries[:10]):
            logger.info(f"   {i+1}. Query: {q}")

        # Convert text queries into vector embeddings
        query_vectors = model.encode(queries).tolist()

        # Store in ChromaDB
        collection.add(
            ids=[str(i) for i in range(len(queries))],
            embeddings=query_vectors,
            metadatas=metadata,
            documents=queries  # Ensures text is retrievable in searches
        )

        logger.info(f"✅ {len(queries)} TMDB queries embedded successfully!")

        # Verify data persistence
        stored_data = collection.get()
        stored_count = len(stored_data.get("ids", []))

        if stored_count > 0:
            logger.info(f"✅ ChromaDB After Insert - Total IDs: {stored_count}")
        else:
            logger.error("❌ ChromaDB After Insert: No queries found! Possible persistence issue.")

    except Exception as e:
        logger.error(f"❌ Error embedding TMDB queries: {e}", exc_info=True)


if __name__ == "__main__":
    embed_tmdb_queries()
