import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tmdb_debug")

# Load TMDB API mappings from JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/tmdb.json")

def load_tmdb_schema():
    """Load TMDB API schema from JSON and extract endpoint details."""
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        api_schema = json.load(file)

    endpoint_mappings = {}

    for path, methods in api_schema.get("paths", {}).items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                logger.warning(f"Skipping malformed entry at {path} - {details}")
                continue

            summary = details.get("summary", f"Query for {path}")
            parameters = details.get("parameters", [])
            
            endpoint_mappings[path] = {
                "method": method.upper(),
                "summary": summary,
                "parameters": parameters
            }

    logger.debug(f"Loaded {len(endpoint_mappings)} API mappings from schema.")
    return endpoint_mappings

# Load the TMDB API map at runtime
TMDB_API_MAP = load_tmdb_schema()

def handle_tmdb_dispatcher(api_call):
    """Handle the TMDB API request based on user query."""
    endpoint = api_call.get("endpoint")
    method = api_call.get("method", "GET")
    parameters = api_call.get("parameters", {})

    if not endpoint:
        logger.error("❌ Error: No valid API endpoint found in request.")
        return {"error": "Invalid API request. Endpoint missing."}

    logger.debug(f"API Call - Endpoint: {endpoint}, Method: {method}, Params: {parameters}")

    # Construct final API request (example format, actual implementation may vary)
    api_url = f"https://api.themoviedb.org/3{endpoint}"
    request_data = {
        "method": method,
        "params": parameters
    }

    return {"api_url": api_url, "request_data": request_data}
