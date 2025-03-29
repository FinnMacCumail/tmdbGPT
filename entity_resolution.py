import requests
from typing import Dict, Optional

class TMDBEntityResolver:
    def __init__(self, api_key: str):
        self.api_key = api_key  # Bearer token
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json;charset=utf-8"
        }
        self.genre_cache = {}
        self._load_genres()

    def _load_genres(self):
        """Dynamically preload genre lists for movies and TV shows"""
        for media_type in ["movie", "tv"]:
            genres = self._fetch_genres(media_type)
            self.genre_cache.update({(media_type, name): genre_id for name, genre_id in genres.items()})

    def _fetch_genres(self, media_type: str) -> Dict[str, int]:
        """Fetch genre mappings from TMDB"""
        try:
            response = requests.get(
                f"{self.base_url}/genre/{media_type}/list",
                headers=self.headers
            )
            response.raise_for_status()
            return {genre["name"].lower(): genre["id"] for genre in response.json().get("genres", [])}
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Failed to fetch {media_type} genres: {e}")
            return {}

    def resolve_entity(self, entity_name: str, entity_type: str, media_type: Optional[str] = None) -> Dict:
        """Resolve entities dynamically, handling genres separately"""
        if entity_type == "genre":
            genre_id = self._resolve_genre(entity_name, media_type)
            return {'id': genre_id, 'name': entity_name, 'type': entity_type} if genre_id else {'id': None, 'name': entity_name, 'type': entity_type}
        else:
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

    def _resolve_genre(self, genre_name: str, media_type: Optional[str]) -> Optional[int]:
        """Resolve genre ID dynamically from cache"""
        genre_name_lower = genre_name.lower()
        if media_type:
            return self.genre_cache.get((media_type, genre_name_lower))
        # Try movie first, then tv as fallback
        return self.genre_cache.get(("movie", genre_name_lower)) or self.genre_cache.get(("tv", genre_name_lower))

    def _make_search_request(self, endpoint: str, query: str) -> Optional[Dict]:
        """Generic search with proper URL encoding and error handling"""
        try:
            params = {
                "query": query,
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