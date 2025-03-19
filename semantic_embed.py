import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Constants
GENRE_MAPPINGS = {
    "movie": {
        28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
        80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
        14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
        9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
        10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
    },
    "tv": {
        10759: 'Action & Adventure', 16: 'Animation', 35: 'Comedy',
        80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
        10762: 'Kids', 9648: 'Mystery', 10763: 'News', 10764: 'Reality',
        10765: 'Sci-Fi & Fantasy', 10766: 'Soap', 10767: 'Talk',
        10768: 'War & Politics', 37: 'Western'
    }
}

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
endpoint_collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./model_cache")

# Load TMDB schema
with open("./data/tmdb.json", "r", encoding="utf-8") as f:
    tmdb_schema = json.load(f)

api_endpoints = tmdb_schema.get("paths", {})

def analyze_parameters(parameters: List) -> Dict:
    """Enhanced parameter analysis with semantic categorization"""
    analysis = {
        "search_params": [],
        "content_filters": [],
        "temporal_filters": [],
        "quality_filters": [],
        "people_entities": [],
        "genre_params": [],
        "pagination_params": [],
        "id_params": [],
        "required_params": [],
        "sort_options": [],
        "provider_filters": []
    }

    for param in parameters:
        if not isinstance(param, dict):
            continue

        param_name = param.get("name", "")
        param_in = param.get("in", "")
        required = param.get("required", False)

        # Parameter classification
        if param_in == "path":
            analysis["id_params"].append(param_name)
        elif param_in == "query":
            if param_name == "query":
                analysis["search_params"].append(param_name)
            elif "page" in param_name:
                analysis["pagination_params"].append(param_name)
            elif any(k in param_name for k in [".gte", ".lte", "date"]):
                analysis["temporal_filters"].append(param_name)
            elif "genre" in param_name:
                analysis["genre_params"].append(param_name)
            elif "vote" in param_name or "rating" in param_name:
                analysis["quality_filters"].append(param_name)
            elif "with_" in param_name:
                if "watch" in param_name:
                    analysis["provider_filters"].append(param_name)
                elif any(k in param_name for k in ["cast", "crew", "people"]):
                    analysis["people_entities"].append(param_name)
            elif "sort_by" in param_name:
                analysis["sort_options"] = param.get("schema", {}).get("enum", [])

        if required:
            analysis["required_params"].append(param_name)

    return analysis

def infer_api_context(path: str, method_data: dict) -> str:
    """Enhanced context inference with path pattern analysis"""
    summary = method_data.get("summary", "").lower()
    parameters = method_data.get("parameters", [])

    # Path-based detection
    if "/discover/" in path:
        return "Advanced content discovery endpoint"
    if "/trending/" in path:
        return "Real-time trending content endpoint"
    if "/similar" in path or "/recommendations" in path:
        return "Content recommendation endpoint"
    if "/credits" in path or "/people" in path:
        return "Cast/Crew information endpoint"

    # Parameter-based detection
    param_names = [p.get("name", "") for p in parameters]
    if "with_genres" in param_names:
        return "Genre-specific content endpoint"
    if "with_cast" in param_names or "with_crew" in param_names:
        return "People-related content endpoint"

    # Summary-based fallback
    if "search" in summary:
        return "Search-oriented endpoint"
    if "discover" in summary:
        return "Content discovery endpoint"
    if "popular" in summary or "top rated" in summary:
        return "Popular content endpoint"

    return "General API endpoint"

def generate_embedding_components(path: str, method: str, method_data: dict) -> Dict[str, Any]:
    """Generates enriched metadata components for embedding"""
    parameters = method_data.get("parameters", [])
    param_analysis = analyze_parameters(parameters)
    context = infer_api_context(path, method_data)

    # Extract genre information
    media_type = "movie" if "/movie" in path else "tv" if "/tv" in path else "other"
    genre_params = [p for p in parameters if "with_genres" in p.get("name", "")]
    available_genres = []
    if genre_params:
        available_genres = list(GENRE_MAPPINGS.get(media_type, {}).values())

    return {
        "path": path,
        "method": method.upper(),
        "summary": method_data.get("summary", "No summary"),
        "context": context,
        "media_type": media_type,
        "parameters": {
            "search_fields": param_analysis["search_params"],
            "content_filters": param_analysis["content_filters"],
            "temporal_filters": param_analysis["temporal_filters"],
            "quality_metrics": param_analysis["quality_filters"],
            "people_entities": param_analysis["people_entities"],
            "available_genres": available_genres,
            "sort_options": param_analysis["sort_options"],
            "provider_filters": param_analysis["provider_filters"],
            "pagination": param_analysis["pagination_params"],
            "id_parameters": param_analysis["id_params"],
            "required_params": param_analysis["required_params"]
        }
    }

def create_embedding_text(components: Dict) -> str:
    """Generates rich semantic text for vector embedding"""
    return (
        f"API Functionality: {components['summary']}\n"
        f"Context: {components['context']}\n"
        f"Media Type: {components['media_type'].upper()}\n"
        f"Path Pattern: {components['path']}\n"
        f"HTTP Method: {components['method']}\n"
        "Capabilities:\n"
        f"- Search Fields: {components['parameters']['search_fields'] or 'None'}\n"
        f"- Content Filters: {components['parameters']['content_filters'] or 'None'}\n"
        f"- Temporal Filters: {components['parameters']['temporal_filters'] or 'None'}\n"
        f"- Quality Metrics: {components['parameters']['quality_metrics'] or 'None'}\n"
        f"- People Entities: {components['parameters']['people_entities'] or 'None'}\n"
        f"- Available Genres: {', '.join(components['parameters']['available_genres']) or 'None'}\n"
        f"- Sort Options: {components['parameters']['sort_options'] or 'None'}\n"
        f"- Provider Filters: {components['parameters']['provider_filters'] or 'None'}\n"
        f"- Pagination: {components['parameters']['pagination'] or 'None'}\n"
        f"- ID Parameters: {components['parameters']['id_parameters'] or 'None'}\n"
        f"- Required Parameters: {components['parameters']['required_params'] or 'None'}"
    )

def store_api_embeddings():
    """Batch process and store enriched embeddings"""
    batch_size = 50
    ids, embeddings, metadatas = [], [], []

    for path, methods in api_endpoints.items():
        for method, method_data in methods.items():
            components = generate_embedding_components(path, method, method_data)
            if "/search/" in path:
                components["summary"] += " search query-based endpoint"
            if "/person/" in path:
                components["summary"] += " individual entity details"
            embedding_text = create_embedding_text(components)
            
            metadata = {
                "path": path,
                "method": components['method'],
                "media_type": components['media_type'],
                "context": components['context'],
                "supports_genre_filtering": bool(components['parameters']['available_genres']),
                "supports_temporal_filtering": bool(components['parameters']['temporal_filters']),
                "supports_actor_search": "with_cast" in components['parameters']['people_entities'],
                "supports_crew_search": "with_crew" in components['parameters']['people_entities'],
                "supports_provider_filtering": bool(components['parameters']['provider_filters']),
                "requires_id_resolution": bool(re.search(r"{\w+}", path))
            }

            ids.append(f"{path}-{method}")
            embeddings.append(model.encode(embedding_text).tolist())
            metadatas.append(metadata)

            # Batch processing
            if len(ids) % batch_size == 0:
                endpoint_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                ids, embeddings, metadatas = [], [], []

    # Add remaining items
    if ids:
        endpoint_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

if __name__ == "__main__":
    store_api_embeddings()