import requests
from typing import Dict, Optional

class TMDBEntityResolver:
    def __init__(self, api_key: str):
        self.api_key = api_key  # This should be a Bearer token
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json;charset=utf-8"
        }
        self._load_genres()

    def _load_genres(self):
        """Preload genre lists for movies and TV shows using headers"""
        self.movie_genres = self._get_genres("movie")
        self.tv_genres = self._get_genres("tv")

    def _get_genres(self, media_type: str) -> Dict[str, int]:
        """Get genres with Bearer authentication"""
        response = requests.get(
            f"{self.base_url}/genre/{media_type}/list",
            headers=self.headers
        )
        response.raise_for_status()
        return {g["name"].lower(): g["id"] for g in response.json().get("genres", [])}

    def resolve_entity(self, entity_name: str, entity_type: str) -> Optional[Dict]:
        """
        Unified entity resolution with proper authentication
        """
        resolver_map = {
            "genre": self._resolve_genre,
            "network": self._resolve_network,
            "credit": lambda x: {"id": x},
            "person": self._resolve_person,
            "movie": self._resolve_movie,
            "tv": self._resolve_tv,
            "company": self._resolve_company,
            "collection": self._resolve_collection,
            "keyword": self._resolve_keyword
        }
        
        if entity_type in resolver_map:
            return resolver_map[entity_type](entity_name)
        return self._resolve_multi(entity_name)

    def _make_search_request(self, endpoint: str, query: str) -> Optional[Dict]:
        """Generic search method with error handling"""
        try:
            params = {
                "query": requests.utils.quote(query),
                "language": "en-US",
                "page": 1
            }
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            return results[0] if results else None
        except requests.exceptions.RequestException as e:
            print(f"API Error: {str(e)}")
            return None
        

    # Simplified resolver methods using the generic search
    def _resolve_genre(self, genre_name: str) -> Optional[Dict]:
        genre_id = (self.movie_genres.get(genre_name.lower()) or 
                    self.tv_genres.get(genre_name.lower()))
        return {"id": genre_id, "name": genre_name} if genre_id else None

    def _resolve_network(self, network_name: str) -> Optional[Dict]:
        return self._make_search_request("/search/company", network_name)

    def _resolve_person(self, person_name: str) -> Optional[Dict]:
        return self._make_search_request("/search/person", person_name)

    def _resolve_movie(self, movie_name: str) -> Optional[Dict]:
        return self._make_search_request("/search/movie", movie_name)

    def _resolve_tv(self, tv_name: str) -> Optional[Dict]:
        return self._make_search_request("/search/tv", tv_name)

    def _resolve_company(self, company_name: str) -> Optional[Dict]:
        return self._make_search_request("/search/company", company_name)

    def _resolve_collection(self, collection_name: str) -> Optional[Dict]:
        return self._make_search_request("/search/collection", collection_name)

    def _resolve_keyword(self, keyword: str) -> Optional[Dict]:
        return self._make_search_request("/search/keyword", keyword)

    def _resolve_multi(self, query: str) -> Optional[Dict]:
        return self._make_search_request("/search/multi", query)
    