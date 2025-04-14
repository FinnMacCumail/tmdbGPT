import requests

class TMDBEntityResolver:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.genre_cache = {
            "movie": self._load_genres("movie"),
            "tv": self._load_genres("tv")
        }

    def _load_genres(self, media_type):
        url = f"{self.base_url}/genre/{media_type}/list"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            genres = response.json().get("genres", [])
            return {g["name"].lower(): g["id"] for g in genres}
        except Exception as e:
            print(f"❌ Failed to load {media_type} genres: {e}")
            return {}

    def resolve_entity(self, name, entity_type):
        if not name or not entity_type:
            return None

        if entity_type == "genre":
            # Try both movie and tv genre caches
            return (
                self.genre_cache["movie"].get(name.lower()) or
                self.genre_cache["tv"].get(name.lower())
            )

        if entity_type in {"person", "movie", "tv", "collection", "company", "keyword", "network"}:
            search_endpoint = f"{self.base_url}/search/{entity_type}"
            try:
                response = requests.get(search_endpoint, headers=self.headers, params={"query": name})
                response.raise_for_status()
                results = response.json().get("results", [])

                # Prefer exact match (case-insensitive)
                for item in results:
                    label = item.get("name") or item.get("title")
                    if label and label.lower() == name.lower():
                        return item.get("id")

                # Fallback to first result
                if results:
                    return results[0].get("id")

            except Exception as e:
                print(f"❌ Failed to resolve entity '{name}' of type '{entity_type}': {e}")

        return None