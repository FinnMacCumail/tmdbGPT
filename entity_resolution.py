import requests
from collections import defaultdict


class TMDBEntityResolver:
    def __init__(self, api_key, headers):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = headers

        self.genre_cache = {
            "movie": self._load_genres("movie"),
            "tv": self._load_genres("tv")
        }

        self.network_cache = {}
        self.company_cache = {}

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
                self.genre_cache["movie"].get(name.lower()) or
                self.genre_cache["tv"].get(name.lower())
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

    def resolve_multiple(self, names, entity_type, top_k=3):
        results = []
        for name in names:
            _id = self.resolve_entity(name, entity_type)
            if _id:
                results.append(_id)
        return results

    def resolve_entities(self, query_entities):
        resolved, unresolved = [], []
        by_type = defaultdict(list)

        # Group entities by type
        for entity in query_entities:
            by_type[entity["type"]].append(entity)

        # Generic TMDB search types
        generic_types = ["person", "movie", "tv", "genre"]
        for entity_type in generic_types:
            for entity in by_type.get(entity_type, []):
                _id = self.resolve_entity(entity["name"], entity_type)
                if _id:
                    entity["resolved_id"] = _id
                    resolved.append(entity)
                else:
                    unresolved.append(entity)

        # Network resolution (with fallback to company)
        unresolved_networks = []
        net_resolved, net_unresolved = self.resolve_networks(by_type.get("network", []))
        resolved.extend(net_resolved)
        unresolved_networks.extend(net_unresolved)

        # Attempt to reclassify failed networks as companies (e.g., Netflix, Amazon Prime)
        for entity in unresolved_networks:
            print(f"↪️ Retrying '{entity['name']}' as company...")
            _id = self.resolve_entity(entity["name"], "company")
            if _id:
                entity["type"] = "company"
                entity["resolved_id"] = _id
                resolved.append(entity)
            else:
                unresolved.append(entity)

        # Company resolution
        comp_resolved, comp_unresolved = self.resolve_companies(by_type.get("company", []))
        resolved.extend(comp_resolved)
        unresolved.extend(comp_unresolved)

        return resolved, unresolved


    def resolve_networks(self, query_entities):
        resolved, unresolved = self._resolve_with_cache(query_entities, "network", self.network_cache)

        for entity in unresolved[:]:
            fallback_resolved, fallback_unresolved = self._retry_entity_as_fallback_type(entity, "company", self.company_cache)
            resolved.extend(fallback_resolved)
            unresolved.remove(entity)
            unresolved.extend(fallback_unresolved)

        return resolved, unresolved

    def resolve_companies(self, query_entities):
        return self._resolve_with_cache(query_entities, "company", self.company_cache)

    def _resolve_with_cache(self, query_entities, entity_type, cache):
        resolved, unresolved = [], []
        for entity in query_entities:
            if entity.get("type") != entity_type or "resolved_id" in entity:
                continue

            name = entity["name"]
            print(f"🌐 Resolving {entity_type}: {name}")

            if name in cache:
                entity["resolved_id"] = cache[name]
                print(f"✅ Cached {entity_type} '{name}' → {cache[name]}")
                resolved.append(entity)
                continue

            try:
                resp = requests.get(
                    f"{self.base_url}/search/{entity_type}",
                    params={"query": name},
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("results"):
                    top_match = data["results"][0]
                    entity["resolved_id"] = top_match["id"]
                    entity["resolved_type"] = entity_type
                    entity["resolved_as"] = "fallback" if entity.get("type") != entity_type else "direct"
                    cache[name] = top_match["id"]
                    print(f"✅ Resolved '{name}' → {top_match['id']}")
                    resolved.append(entity)
                else:
                    print(f"❌ No results for {entity_type} '{name}'")
                    unresolved.append(entity)
            except Exception as e:
                print(f"❌ Error resolving {entity_type} '{name}': {e}")
                unresolved.append(entity)

        return resolved, unresolved
    
    def _retry_entity_as_fallback_type(self, entity, fallback_type, cache):
        name = entity["name"]
        print(f"↪️ Retrying '{name}' as {fallback_type}...")

        if name in cache:
            entity["resolved_id"] = cache[name]
            print(f"✅ Cached fallback {fallback_type} '{name}' → {cache[name]}")
            return [entity], []

        try:
            resp = requests.get(
                f"{self.base_url}/search/{fallback_type}",
                params={"query": name},
                headers=self.headers
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                entity["resolved_id"] = results[0]["id"]
                cache[name] = results[0]["id"]
                print(f"✅ Resolved fallback '{name}' as {fallback_type} → {results[0]['id']}")
                return [entity], []
            else:
                return [], [entity]
        except Exception as e:
            print(f"❌ Fallback failed for {fallback_type} '{name}': {e}")
            return [], [entity]
