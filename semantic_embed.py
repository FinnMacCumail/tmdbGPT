import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from collections import defaultdict

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
    """Enhanced parameter analysis with relationship detection"""
    analysis = defaultdict(list)
    analysis.update({
        "search_params": [],
        "content_filters": [],
        "temporal_filters": [],
        "quality_filters": [],
        "relationship_filters": [],
        "genre_params": [],
        "pagination_params": [],
        "id_params": [],
        "required_params": [],
        "sort_options": [],
        "provider_filters": []
    })

    for param in parameters:
        if not isinstance(param, dict):
            continue

        pname = param.get("name", "").lower()
        param_in = param.get("in", "").lower()
        required = param.get("required", False)

        if param_in == "path":
            analysis["id_params"].append(pname)
        elif param_in == "query":
            if pname == "query":
                analysis["search_params"].append(pname)
            elif "page" in pname:
                analysis["pagination_params"].append(pname)
            elif any(k in pname for k in [".gte", ".lte", "date"]):
                analysis["temporal_filters"].append(pname)
            elif "genre" in pname:
                analysis["genre_params"].append(pname)
            elif "vote" in pname or "rating" in pname:
                analysis["quality_filters"].append(pname)
            elif "with_" in pname:
                if "watch" in pname:
                    analysis["provider_filters"].append(pname)
                elif any(k in pname for k in ["cast", "crew", "people"]):
                    analysis["relationship_filters"].append(pname)
            elif "sort_by" in pname:
                analysis["sort_options"] = param.get("schema", {}).get("enum", [])

        if required:
            analysis["required_params"].append(pname)

    return analysis

def _get_operation_type(path: str, params: Dict) -> str:
    """Hierarchical endpoint categorization"""
    path = path.lower()
    
    if "/search/" in path:
        return "Entity Discovery"
    if "/discover/" in path:
        return "Complex Filtering"
    if "/trending/" in path or "popular" in path:
        return "Trending Content"
    if "/credits" in path or "/crew" in path:
        return "Relationship Mapping"
    if any(k in path for k in ["/images", "/videos", "/posters"]):
        return "Media Asset Retrieval"
    if "/list" in path or "genre" in path:
        return "Static Data Access"
        
    if "/{" in path:
        if any(k in path for k in ["/similar", "/recommendations"]):
            return "Content Recommendations"
        if "reviews" in path:
            return "User Feedback"
        return "Core Entity Details"
    
    if params["search_params"]:
        return "Entity Discovery"
    if params["sort_options"]:
        return "Sorted Collection"
        
    return "General Access"

def _get_search_targets(components: Dict) -> List[str]:
    """Identify what entities this endpoint can help resolve"""
    if components['parameters']['search_params']:
        media_type = components['media_type']
        if media_type in ['movie', 'tv', 'person']:
            return [media_type]
    return []

def generate_embedding_components(path: str, method: str, method_data: dict) -> Dict[str, Any]:
    """Generates enriched metadata components with search capabilities"""
    parameters = method_data.get("parameters", [])
    param_analysis = analyze_parameters(parameters)
    
    # Determine media type with person support
    media_type = "other"
    if "/movie" in path:
        media_type = "movie"
    elif "/tv" in path:
        media_type = "tv"
    elif "/person" in path:
        media_type = "person"

    components = {
        "path": path,
        "method": method.upper(),
        "summary": method_data.get("summary", "No summary"),
        "operation_type": _get_operation_type(path, param_analysis),
        "media_type": media_type,
        "parameters": param_analysis,
        "resolution_dependencies": list(set(re.findall(r"{(\w+_id)}", path))),
        "available_genres": list(GENRE_MAPPINGS.get(media_type, {}).values()),
        "search_capable_for": _get_search_targets({
            'parameters': param_analysis,
            'media_type': media_type
        })
    }
    return components

def create_embedding_text(components: Dict) -> str:
    """Generates structured semantic text for embeddings"""
    return (
        f"API Function: {components['summary']}\n"
        f"Operation Type: {components['operation_type']}\n"
        f"Media Focus: {components['media_type'].upper()}\n"
        f"Search Capabilities: {', '.join(components['search_capable_for']) or 'None'}\n"
        f"Requires IDs: {', '.join(components['resolution_dependencies']) or 'None'}\n"
        f"Path Structure: {components['path']}\n"
        f"Supported Filters: {_format_filter_types(components)}\n"
        f"Complexity Level: {len(components['parameters']['required_params'])}"
    )

def _format_filter_types(components: Dict) -> str:
    """Formats filter capabilities for embedding text"""
    filters = []
    if components['parameters']['temporal_filters']:
        filters.append("Temporal")
    if components['parameters']['genre_params']:
        filters.append("Genre")
    if components['parameters']['relationship_filters']:
        filters.append("Relationships")
    return ', '.join(filters) or "None"

def store_api_embeddings():
    """Batch process and store enhanced embeddings with Chroma-compatible metadata"""
    batch_size = 50
    ids, embeddings, metadatas = [], [], []

    for path, methods in api_endpoints.items():
        for method, method_data in methods.items():
            components = generate_embedding_components(path, method, method_data)
            
            # Convert list-based fields to comma-separated strings
            metadata = {
                "path": path,
                "method": components['method'],
                "operation_type": components['operation_type'],
                "media_type": components['media_type'],
                "search_capable_for": ",".join(components['search_capable_for']),
                "resolution_dependencies": ",".join(components['resolution_dependencies']),
                "filter_temporal": str(bool(components['parameters']['temporal_filters'])),
                "filter_genre": str(bool(components['parameters']['genre_params'])),
                "filter_relationships": str(bool(components['parameters']['relationship_filters'])),
                "complexity_score": len(components['parameters']['required_params']),
                "requires_id_resolution": str(bool(components['resolution_dependencies']))
            }

            embedding_text = create_embedding_text(components)
            
            ids.append(f"{path}-{method}")
            embeddings.append(model.encode(embedding_text).tolist())
            metadatas.append(metadata)

            if len(ids) % batch_size == 0:
                endpoint_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                ids, embeddings, metadatas = [], [], []

    if ids:
        endpoint_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        
if __name__ == "__main__":
    store_api_embeddings()