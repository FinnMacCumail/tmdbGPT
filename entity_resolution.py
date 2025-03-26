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

    def resolve_entity(self, entity_name: str, entity_type: str) -> Dict:
        """Ensure resolved IDs are integers"""
        result = self._make_search_request(f"/search/{entity_type}", entity_name)
        
        if result and 'id' in result:
            try:
                return {
                    'id': int(result['id']),
                    'name': result.get('name', entity_name),
                    'type': entity_type
                }
            except (ValueError, TypeError):
                print(f"🚨 Invalid ID format for {entity_type}: {result['id']}")
        
        return {'id': None, 'name': entity_name, 'type': entity_type}

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
