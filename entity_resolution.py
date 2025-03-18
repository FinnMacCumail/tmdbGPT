import requests
from typing import Dict, Optional
import json

class TMDBEntityResolver:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self._load_genres()

    def _load_genres(self):
        # Preload genre lists for movies and TV shows once
        self.movie_genres = self._get_genres("movie")
        self.tv_genres = self._get_genres("tv")

    def _get_genres(self, media_type: str) -> Dict[str, int]:
        response = requests.get(
            f"{self.base_url}/genre/{media_type}/list",
            params={"api_key": self.api_key}
        )
        genres = response.json().get("genres", [])
        return {g["name"].lower(): g["id"] for g in genres}

    def resolve_entity(self, entity_name: str, entity_type: str) -> Optional[Dict]:
        """
        Resolves an entity based on its type.
        For genres, it checks the preloaded lists.
        For credits, it returns the provided value.
        For other types, it uses the corresponding TMDB search endpoint.
        """
        if entity_type == "genre":
            return self._resolve_genre(entity_name)
        elif entity_type == "network":
            return self._resolve_network(entity_name)
        elif entity_type == "credit":
            return {"id": entity_name}  # Credits use direct IDs
        elif entity_type == "person":
            return self._resolve_person(entity_name)
        elif entity_type == "movie":
            return self._resolve_movie(entity_name)
        elif entity_type == "tv":
            return self._resolve_tv(entity_name)
        elif entity_type == "company":
            return self._resolve_company(entity_name)
        elif entity_type == "collection":
            return self._resolve_collection(entity_name)
        elif entity_type == "keyword":
            return self._resolve_keyword(entity_name)
        else:
            # Fallback to multi-search if type is unrecognized
            return self._resolve_multi(entity_name)

    def _resolve_genre(self, genre_name: str) -> Optional[Dict]:
        genre_id = (self.movie_genres.get(genre_name.lower()) or 
                    self.tv_genres.get(genre_name.lower()))
        return {"id": genre_id, "name": genre_name} if genre_id else None

    def _resolve_network(self, network_name: str) -> Optional[Dict]:
        params = {
            "api_key": self.api_key,
            "query": network_name,
            "language": "en-US"
        }
        response = requests.get(f"{self.base_url}/search/company", params=params)
        results = response.json().get("results", [])
        return results[0] if results else None

    def _resolve_person(self, person_name: str) -> Optional[Dict]:
        params = {
            "api_key": self.api_key,
            "query": person_name,
            "language": "en-US"
        }
        response = requests.get(f"{self.base_url}/search/person", params=params)
        results = response.json().get("results", [])
        return results[0] if results else None

    def _resolve_movie(self, movie_name: str) -> Optional[Dict]:
        params = {
            "api_key": self.api_key,
            "query": movie_name,
            "language": "en-US"
        }
        response = requests.get(f"{self.base_url}/search/movie", params=params)
        results = response.json().get("results", [])
        return results[0] if results else None

    def _resolve_tv(self, tv_name: str) -> Optional[Dict]:
        params = {
            "api_key": self.api_key,
            "query": tv_name,
            "language": "en-US"
        }
        response = requests.get(f"{self.base_url}/search/tv", params=params)
        results = response.json().get("results", [])
        return results[0] if results else None

    def _resolve_company(self, company_name: str) -> Optional[Dict]:
        params = {
            "api_key": self.api_key,
            "query": company_name,
            "language": "en-US"
        }
        response = requests.get(f"{self.base_url}/search/company", params=params)
        results = response.json().get("results", [])
        return results[0] if results else None

    def _resolve_collection(self, collection_name: str) -> Optional[Dict]:
        params = {
            "api_key": self.api_key,
            "query": collection_name,
            "language": "en-US"
        }
        response = requests.get(f"{self.base_url}/search/collection", params=params)
        results = response.json().get("results", [])
        return results[0] if results else None

    def _resolve_keyword(self, keyword: str) -> Optional[Dict]:
        params = {
            "api_key": self.api_key,
            "query": keyword,
            "language": "en-US"
        }
        response = requests.get(f"{self.base_url}/search/keyword", params=params)
        results = response.json().get("results", [])
        return results[0] if results else None

    def _resolve_multi(self, query: str) -> Optional[Dict]:
        # Generic multi-search if the entity type isn’t explicitly handled
        params = {
            "api_key": self.api_key,
            "query": query,
            "language": "en-US"
        }
        response = requests.get(f"{self.base_url}/search/multi", params=params)
        results = response.json().get("results", [])
        return results[0] if results else None
