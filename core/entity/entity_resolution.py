# entity_resolution.py

import requests
from datetime import datetime, timedelta
import os
from core.entity.param_utils import GenreNormalizer
import json
from typing import Optional


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
        for media_type in ["movie", "tv"]:
            url = f"{self.base_url}/genre/{media_type}/list"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                genres = response.json().get("genres", [])
                self.genre_cache[media_type] = {
                    g["name"]: g["id"] for g in genres}
            # else:
        self.genre_cache_timestamp = datetime.now()

    def _resolve_genre_id(self, genre_name: str, intended_media_type: str = "movie") -> int:
        self._refresh_genre_cache()

        # ✅ Use correct media_type
        canonical_name = GenreNormalizer.normalize(
            genre_name, intended_media_type)
        genre_map = self.genre_cache.get(intended_media_type, {})

        for name, gid in genre_map.items():
            if canonical_name.lower() in name.lower():
                return gid

        # fallback to movie genres if not found
        if intended_media_type != "movie":
            for name, gid in self.genre_cache.get("movie", {}).items():
                if canonical_name.lower() in name.lower():
                    # Debug output removed
                    return gid


        return None

    def resolve_entity(self, name: str, entity_type: str) -> Optional[int]:
        name_normalized = name.strip().lower()
        
        # Normalize common company/network name variations
        name_variations = [name_normalized]
        
        # Handle plural/singular forms
        if name_normalized.endswith('s') and len(name_normalized) > 3:
            name_variations.append(name_normalized[:-1])  # Remove 's'
        else:
            name_variations.append(name_normalized + 's')   # Add 's'
        
        # Handle common company suffixes  
        suffixes_to_try = ['', ' entertainment', ' entertainments', ' studios', ' studio', 
                          ' pictures', ' films', ' productions']
        base_name = name_normalized
        for suffix in [' entertainment', ' entertainments', ' studios', ' studio', 
                      ' pictures', ' films', ' productions']:
            if name_normalized.endswith(suffix):
                base_name = name_normalized.replace(suffix, '')
                break
        
        for suffix in suffixes_to_try:
            variant = base_name + suffix
            if variant not in name_variations:
                name_variations.append(variant)

        # Special handling for networks using local cache
        if entity_type == "network":
            if hasattr(self, 'network_cache'):
                # Special BBC handling - override cache for generic 'bbc' to prefer UK BBC One
                if name_normalized == "bbc":
                    # Update cache to prefer UK BBC One over Japanese BBC
                    self.network_cache[name_normalized] = 4
                    return 4
                
                if name_normalized in self.network_cache:
                    network_id = self.network_cache[name_normalized]
                    # Debug output removed
                    return network_id

                for cached_name, nid in self.network_cache.items():
                    if name_normalized in cached_name or cached_name in name_normalized:
                        # Debug output removed
                        return nid

        try:
            response = requests.get(
                f"{self.base_url}/search/{entity_type}",
                headers=self.headers,
                params={"query": name},
            )
            response.raise_for_status()
            results = response.json().get("results", [])

            if entity_type == "company":
                # Company-specific matching logic with US preference
                # Handle company name variations and international versions
                us_preferred_companies = {"legendary entertainment", "legendary", "marvel", "disney", 
                                        "warner bros", "universal", "paramount", "sony", "netflix"}
                needs_us_preference = any(pref in name_normalized for pref in us_preferred_companies)
                
                if needs_us_preference:
                    # Strongly prefer US versions for major studios
                    for item in results:
                        label = item.get("name", "").strip().lower()
                        for name_var in name_variations:
                            if label == name_var and item.get("origin_country") == "US":
                                return item["id"]
                    
                    for item in results:
                        label = item.get("name", "").strip().lower()
                        for name_var in name_variations:
                            if name_var in label and item.get("origin_country") == "US":
                                return item["id"]
                else:
                    # Standard company matching with US preference
                    for item in results:
                        label = item.get("name", "").strip().lower()
                        if label == name_normalized and item.get("origin_country") == "US":
                            return item["id"]

                    for item in results:
                        label = item.get("name", "").strip().lower()
                        if name_normalized in label and item.get("origin_country") == "US":
                                # Debug output removed
                            return item["id"]

            elif entity_type == "network":
                # Network-specific matching logic with origin preference
                
                # BBC should prefer UK/GB origin 
                if "bbc" in name_normalized:
                    # First try for BBC One specifically (ID: 4) - most popular BBC network
                    if name_normalized in {"bbc", "bbc one", "bbc 1"}:
                        return 4  # Direct return for BBC One
                    
                    # For other BBC variants, prefer GB origin
                    for item in results:
                        label = item.get("name", "").strip().lower()
                        origin = item.get("origin_country", "")
                        if "bbc" in label and origin == "GB":
                            return item["id"]
                    
                    # Fallback for BBC - still prefer GB but allow exact matches
                    for item in results:
                        label = item.get("name", "").strip().lower()
                        if label == name_normalized and item.get("origin_country") == "GB":
                            return item["id"]
                
                # Strong US preference for major international services
                us_preferred_networks = {"hbo", "netflix", "hulu", "amazon prime", "disney+", 
                                       "apple tv", "peacock", "paramount+", "starz", "showtime"}
                needs_us_preference = name_normalized in us_preferred_networks
                
                if needs_us_preference:
                    # Strongly prefer US versions for major services
                    for item in results:
                        label = item.get("name", "").strip().lower()
                        for name_var in name_variations:
                            if label == name_var and item.get("origin_country") == "US":
                                return item["id"]
                    
                    for item in results:
                        label = item.get("name", "").strip().lower()
                        for name_var in name_variations:
                            if name_var in label and item.get("origin_country") == "US":
                                return item["id"]
                else:
                    # Standard network matching with US preference
                    for item in results:
                        label = item.get("name", "").strip().lower()
                        if label == name_normalized and item.get("origin_country") == "US":
                            return item["id"]

                    for item in results:
                        label = item.get("name", "").strip().lower()
                        if name_normalized in label and item.get("origin_country") == "US":
                            return item["id"]

            # Generic exact match for all types
            for item in results:
                label = (item.get("name") or item.get(
                    "title") or "").strip().lower()
                if label == name_normalized:
                        # Debug output removed
                    return item["id"]

            # Generic substring match for all types
            for item in results:
                label = (item.get("name") or item.get(
                    "title") or "").strip().lower()
                if name_normalized in label:
                        # Debug output removed
                    return item["id"]

            # Fallback to first result
            if results:
                fallback = results[0]
                    # Debug output removed
                return fallback["id"]

        except Exception as e:
            # Debug output removed
            pass
            
        return None

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
                    entity["type"] = "network"
                else:
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
                unresolved_entities.append(entity)

        return resolved_entities, unresolved_entities

    def _load_offline_network_cache(self):
        path = "data/tv_network_ids_05_01_2025.json"
        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    name = entry.get("name", "").strip().lower()
                    nid = entry.get("id")
                    if name and nid:
                        self.network_cache[name] = nid
        except Exception as e:
            pass
