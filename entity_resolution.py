import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional, List
import requests
from functools import lru_cache
import json

# Constants
TMDB_API_KEY = "your_tmdb_api_key"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Initialize ChromaDB for entity resolution
chroma_client = chromadb.PersistentClient(path="./chroma_db")
entity_collection = chroma_client.get_or_create_collection(
    name="tmdb_entities",
    metadata={"hnsw:space": "cosine"}
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./model_cache")

class TMDBEntityResolver:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _search_tmdb(self, entity_name: str, entity_type: str) -> Optional[Dict]:
        """Search TMDB API for an entity."""
        endpoint = {
            "person": "/search/person",
            "movie": "/search/movie",
            "tv": "/search/tv",
            "company": "/search/company",
            "collection": "/search/collection",
            "keyword": "/search/keyword"
        }.get(entity_type, "/search/multi")

        params = {
            "api_key": self.api_key,
            "query": entity_name,
            "language": "en-US"
        }

        response = requests.get(f"{TMDB_BASE_URL}{endpoint}", params=params)
        if response.status_code == 200:
            results = response.json().get("results", [])
            return results[0] if results else None
        return None

    def resolve_entity(self, entity_name: str, entity_type: str) -> Optional[Dict]:
        """
        Resolves an entity to its TMDB ID and metadata.
        First checks ChromaDB, then falls back to TMDB API.
        """
        # Check ChromaDB first
        results = entity_collection.query(
            query_texts=[entity_name],
            where={"type": entity_type},
            n_results=1
        )
        if results["ids"]:
            return json.loads(results["documents"][0])

        # Fallback to TMDB API
        entity_data = self._search_tmdb(entity_name, entity_type)
        if entity_data:
            self._add_entity_to_chroma(entity_data, entity_type)
            return entity_data
        return None

    def _add_entity_to_chroma(self, entity_data: Dict, entity_type: str):
        """Adds an entity to the ChromaDB collection."""
        embedding_text = self._create_embedding_text(entity_data, entity_type)
        metadata = {
            "type": entity_type,
            "popularity": entity_data.get("popularity", 0),
            "known_for": entity_data.get("known_for_department", ""),
            "tmdb_id": entity_data.get("id")
        }

        entity_collection.add(
            ids=[f"{entity_type}_{entity_data['id']}"],
            embeddings=[model.encode(embedding_text).tolist()],
            metadatas=[metadata],
            documents=[json.dumps(entity_data)]
        )

    def _create_embedding_text(self, entity_data: Dict, entity_type: str) -> str:
        """Generates embedding text for an entity."""
        return (
            f"Entity: {entity_data.get('name', entity_data.get('title'))}\n"
            f"Type: {entity_type}\n"
            f"Known For: {entity_data.get('known_for_department', 'N/A')}\n"
            f"Popularity: {entity_data.get('popularity', 0)}\n"
            f"Description: {entity_data.get('overview', 'No description available')}"
        )

# Example usage
if __name__ == "__main__":
    resolver = TMDBEntityResolver(TMDB_API_KEY)

    # Resolve an entity (e.g., "Keanu Reeves")
    entity = resolver.resolve_entity("Keanu Reeves", "person")
    if entity:
        print(f"Resolved Entity: {entity['name']} (ID: {entity['id']})")
    else:
        print("Entity not found.")