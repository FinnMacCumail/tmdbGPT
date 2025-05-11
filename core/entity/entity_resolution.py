# entity_resolution.py

import requests
from datetime import datetime, timedelta
import os
from core.entity.param_utils import GenreNormalizer
import json


class TMDBEntityResolver:
    def __init__(self, api_key, headers):
        self.api_key = api_key
        self.headers = headers
        self.base_url = "https://api.themoviedb.org/3"
        self.genre_cache = {"movie": {}, "tv": {}}
        self.genre_cache_timestamp = None
        self.network_cache = {}  # new
        self.network_cache_timestamp = None

        self._load_offline_network_cache()

    def _refresh_genre_cache(self):
        if self.genre_cache["movie"] and self.genre_cache["tv"] and self.genre_cache_timestamp:
            if datetime.now() - self.genre_cache_timestamp < timedelta(hours=24):
                return
        # print("🔄 Refreshing TMDB genre cache...")
        for media_type in ["movie", "tv"]:
            url = f"{self.base_url}/genre/{media_type}/list"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                genres = response.json().get("genres", [])
                self.genre_cache[media_type] = {
                    g["name"]: g["id"] for g in genres}
                # print(f"✅ Loaded {len(genres)} {media_type} genres.")
            # else:
            #     print(f"⚠️ Failed to fetch {media_type} genres (status {response.status_code})")
        self.genre_cache_timestamp = datetime.now()

    def _resolve_genre_id(self, genre_name: str, intended_media_type: str = "movie") -> int:
        self._refresh_genre_cache()

        # ✅ Use correct media_type
        canonical_name = GenreNormalizer.normalize(
            genre_name, intended_media_type)
        # print(f"🎯 After normalization: '{genre_name}' → '{canonical_name}' for media_type={intended_media_type}")
        genre_map = self.genre_cache.get(intended_media_type, {})

        for name, gid in genre_map.items():
            if canonical_name.lower() in name.lower():
                return gid

        # fallback to movie genres if not found
        if intended_media_type != "movie":
            for name, gid in self.genre_cache.get("movie", {}).items():
                if canonical_name.lower() in name.lower():
                    print(
                        f"⚠️ Fallback match in movie genres for '{canonical_name}'.")
                    return gid

        # print(f"🔎 _resolve_genre_id got genre_name='{genre_name}', intended_media_type='{intended_media_type}'")

        # print(f"⚠️ No genre ID found for '{canonical_name}' with media_type={intended_media_type}")
        return None

    def resolve_entity(self, name: str, entity_type: str) -> int:
        if entity_type in {"person", "movie", "tv", "collection", "company", "keyword"}:
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
                        # print(f"✅ Resolved {entity_type} '{name}' → {item.get('id')}")
                        return item.get("id")

                if results:
                    fallback_id = results[0].get("id")
                    # print(f"⚠️ Fallback resolution for {entity_type} '{name}' → {fallback_id}")
                    return fallback_id

                # print(f"❌ No results for {entity_type} '{name}'")
            except Exception as e:
                print(
                    f"❌ Failed to resolve entity '{name}' of type '{entity_type}': {e}")

        elif entity_type == "network":
            lower_name = name.strip().lower()

            # ✅ 1. Exact match from local cache
            if lower_name in self.network_cache:
                network_id = self.network_cache[lower_name]
                # print(f"✅ Resolved network '{name}' → {network_id} (from local file)")
                return network_id

            # ✅ 2. Fuzzy match fallback
            for cached_name, nid in self.network_cache.items():
                if lower_name in cached_name or cached_name in lower_name:
                    # print(f"⚡ Fuzzy matched network '{name}' → '{cached_name}' → {nid}")
                    return nid

    def resolve_entities(self, query_entities, intended_media_type="movie"):

        dynamic_services = {"netflix", "amazon prime", "prime video",
                            "hulu", "disney+", "apple tv", "peacock", "paramount+"}
        always_network_services = {"hbo", "starz"}

        for entity in query_entities:
            ent_type = entity.get("type")
            name = entity.get("name", "")

            name_lower = name.lower().strip()

            # --- 📦 Dynamic correction ---
            if name_lower in dynamic_services:
                if intended_media_type == "tv":
                    # print(f"🔁 Correcting '{name}' type dynamically during resolution: TV detected → network")
                    entity["type"] = "network"
                else:
                    # print(f"🔁 Correcting '{name}' type dynamically during resolution: Movie detected → company")
                    entity["type"] = "company"
                entity["resolved_as"] = "dynamic"
            elif name_lower in always_network_services:
                entity["type"] = "network"
                entity["resolved_as"] = "static"

        resolved_entities = []
        unresolved_entities = []

        for entity in query_entities:
            ent_type = entity.get("type")
            name = entity.get("name")

            if ent_type == "genre":
                # ✅ Always use the passed intended_media_type
                resolved_id = self._resolve_genre_id(name, intended_media_type)
                if resolved_id:
                    entity["resolved_id"] = resolved_id
                    entity["resolved_type"] = "genre"
                    resolved_entities.append(entity)
                    # print(f"🎯 Resolved genre '{name}' → {resolved_id} for {intended_media_type}")
                else:
                    unresolved_entities.append(entity)

            elif ent_type in {"person", "movie", "tv", "collection", "company", "keyword", "network"}:
                resolved_id = self.resolve_entity(name, ent_type)
                if resolved_id:
                    entity["resolved_id"] = resolved_id
                    entity["resolved_type"] = ent_type
                    resolved_entities.append(entity)
                else:
                    unresolved_entities.append(entity)

            else:
                # print(f"⚠️ Unknown entity type '{ent_type}' for entity '{name}'.")
                unresolved_entities.append(entity)

        return resolved_entities, unresolved_entities

    def _load_offline_network_cache(self):
        path = "data/tv_network_ids_05_01_2025.json"
        if not os.path.exists(path):
            # print("⚠️ Network cache file not found.")
            return

        # print(f"📥 Loading TMDB networks from {path}...")
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    name = entry.get("name", "").strip().lower()
                    nid = entry.get("id")
                    if name and nid:
                        self.network_cache[name] = nid
            # print(f"✅ Loaded {len(self.network_cache)} networks into cache.")
        except Exception as e:
            print(f"❌ Failed to load network cache: {e}")
