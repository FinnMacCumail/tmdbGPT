import requests
from collections import defaultdict


class TMDBEntityResolver:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.genre_cache = {
            "movie": self._load_genres("movie"),
            "tv": self._load_genres("tv"),
        }
        self.network_cache = {}

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
            return (
                self.genre_cache["movie"].get(name.lower())
                or self.genre_cache["tv"].get(name.lower())
            )

        if entity_type in {"person", "movie", "tv", "collection", "company", "keyword", "network"}:
            try:
                response = requests.get(
                    f"{self.base_url}/search/{entity_type}",
                    headers=self.headers,
                    params={"query": name},
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                for item in results:
                    label = item.get("name") or item.get("title")
                    if label and label.lower() == name.lower():
                        return item.get("id")
                if results:
                    return results[0].get("id")
            except Exception as e:
                print(f"❌ Failed to resolve entity '{name}' of type '{entity_type}': {e}")

        return None

    def resolve_multiple(self, names: list, entity_type: str, top_k: int = 3) -> list:
        results = []
        for name in names:
            _id = self.resolve_entity(name, entity_type)
            if _id:
                results.append(_id)
        return results

    def resolve_entities(self, query_entities):
        resolved, unresolved = [], []
        by_type = defaultdict(list)

        for entity in query_entities:
            by_type[entity["type"]].append(entity)

        for entity_type in ["person", "movie", "tv", "genre"]:
            for entity in by_type.get(entity_type, []):
                _id = self.resolve_entity(entity["name"], entity_type)
                if _id:
                    entity["resolved_id"] = _id
                    resolved.append(entity)
                else:
                    unresolved.append(entity)

        net_resolved, net_unresolved = self.resolve_networks(by_type.get("network", []))
        resolved.extend(net_resolved)
        unresolved.extend(net_unresolved)

        return resolved, unresolved

    def resolve_networks(self, query_entities):
        resolved, unresolved = [], []

        for entity in query_entities:
            if entity.get("type") == "network" and "resolved_id" not in entity:
                name = entity["name"]

                if name in self.network_cache:
                    entity["resolved_id"] = self.network_cache[name]
                    resolved.append(entity)
                    continue

                try:
                    resp = requests.get(
                        f"{self.base_url}/search/network",
                        params={"query": name},
                        headers=self.headers,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("results"):
                        top_match = data["results"][0]
                        entity["resolved_id"] = top_match["id"]
                        self.network_cache[name] = top_match["id"]
                        resolved.append(entity)
                    else:
                        unresolved.append(entity)
                except:
                    unresolved.append(entity)

        return resolved, unresolved
