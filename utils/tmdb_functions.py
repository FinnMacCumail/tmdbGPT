import json
import os
from utils.logger import get_logger

logger = get_logger("tmdb_functions")

# Load TMDB API schema
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/tmdb.json")

def load_tmdb_schema():
    """Load TMDB OpenAPI schema from JSON."""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as file:
            api_schema = json.load(file)
        logger.info(f"✅ Successfully loaded TMDB schema from {DATA_PATH}")
        return api_schema
    except Exception as e:
        logger.error(f"❌ Error loading TMDB schema: {e}")
        return {}

def extract_parameters_from_list(param_list):
    """Extract parameters from API schema."""
    return {item["name"]: item.get("description", "No description available") for item in param_list if isinstance(item, dict)}
