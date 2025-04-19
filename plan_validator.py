import json
from chromadb import PersistentClient

class PlanValidator:
    def __init__(self):
        self.client = PersistentClient(path="./chroma_db")
        self.param_collection = self.client.get_or_create_collection("tmdb_parameters")
        self.PARAM_USED_IN = {}
        self._preload_parameter_usage()

    def _preload_parameter_usage(self):
        results = self.param_collection.get(include=["metadatas"])
        for meta in results["metadatas"]:
            name = meta.get("name")
            used_in = meta.get("used_in", [])
            if isinstance(used_in, str):
                try:
                    used_in = json.loads(used_in)
                except:
                    used_in = []
            self.PARAM_USED_IN[name] = used_in

    def _resolve_required_parameters_from_entities(self, query_entities):
        ENTITY_PARAM_MAP = {
            "person": "with_people",
            "movie": "with_movies",
            "tv": "with_tv",
            "genre": "with_genres",
            "company": "with_companies",
            "keyword": "with_keywords",
            "collection": "with_collections",
            "network": "with_networks",
            "review": "with_reviews",
            "credit": "with_credits",
            "language": "with_original_language",
            "country": "region",
            "rating": "certification.gte",
            "date": "primary_release_date.gte"
        }
        return list({
            ENTITY_PARAM_MAP.get(ent["type"])
            for ent in query_entities
            if ENTITY_PARAM_MAP.get(ent["type"])
        })

    def _endpoint_supports_required_params(self, endpoint_path, required_params):
        for param in required_params:
            supported_paths = self.PARAM_USED_IN.get(param, [])
            if endpoint_path not in supported_paths:
                return False
        return True

    def resolve_path_slots(self, query_entities=None, entities=None, intents=None):
        query_entities = query_entities or []
        entity_types = set()

        if query_entities:
            entity_types = {e["type"] for e in query_entities if isinstance(e, dict)}
        elif entities:
            entity_types = {e.replace("_id", "") for e in entities}

        PATH_PARAM_SLOT_MAP = {
            "movie": "media_type",
            "tv": "media_type",
            "date": "time_window"
        }

        path_params = {}

        # ✅ Priority 1: Explicit entity-to-path slot mapping
        if "tv" in entity_types:
            path_params["media_type"] = "tv"
        elif "movie" in entity_types:
            path_params["media_type"] = "movie"

        if "date" in entity_types:
            path_params["time_window"] = "week"

        # ✅ Priority 2: Intent-based fallback if slot still missing
        if intents and "trending.popular" in intents:
            if "media_type" not in path_params:
                path_params["media_type"] = "movie"
            if "time_window" not in path_params:
                path_params["time_window"] = "week"

        return path_params


    def inject_path_slot_parameters(self, step, resolved_entities, extraction_result=None):
        query_entities = extraction_result.get("query_entities", []) if extraction_result else []
        # ✅ FIXED: use LLM-detected entities instead of resolved entity keys
        entities = extraction_result.get("entities", []) if extraction_result else []
        intents = extraction_result.get("intents", []) if extraction_result else []

        path_slots = self.resolve_path_slots(
            query_entities=query_entities,
            entities=entities,
            intents=intents
        )

        for slot, value in path_slots.items():
            if f"{{{slot}}}" in step["endpoint"] and slot not in step["parameters"]:
                step["parameters"][slot] = value
                print(f"🧩 Injected path slot: {slot} = {value} into {step['endpoint']}")

        print(f"✅ Final injected parameters: {step['parameters']}")
        return step
    
    def validate(self, semantic_matches, state):
        query_entities = []

        if hasattr(state, "extraction_result"):
            result = getattr(state, "extraction_result")
            if isinstance(result, dict):
                query_entities = result.get("query_entities", [])
            elif hasattr(result, "query_entities"):
                query_entities = result.query_entities

        required_params = self._resolve_required_parameters_from_entities(query_entities)
        print(f"🔍 Resolved required parameters: {required_params}")

        if not required_params:
            print("🔎 No symbolic filtering required — using all semantic matches.")
            return semantic_matches

        filtered_matches = []
        for m in semantic_matches:
            path = m["path"]
            if self._endpoint_supports_required_params(path, required_params):
                print(f"✅ Included: {path}")
                filtered_matches.append(m)
            else:
                print(f"❌ Excluded: {path} — missing one of: {required_params}")

        if not filtered_matches:
            print("⚠️ No strict param-compatible endpoints found, falling back to semantic matches.")
            return semantic_matches

        return filtered_matches

class SymbolicConstraintFilter:
    @staticmethod
    def apply(matches: list, extraction_result: dict, resolved_entities: dict) -> list:
        """
        General symbolic filtering based on:
        - Entity compatibility (resolved entity must be consumable by endpoint)
        - Media type consistency (tv/movie intent should match endpoint media_type)
        - Optional: Intent compatibility (e.g., discovery.filtered only matches discover endpoints)
        """
        entities = extraction_result.get("entities", [])
        query_intents = extraction_result.get("intents", [])
        media_pref = SymbolicConstraintFilter._infer_media_preference(entities)
        resolved_keys = set(resolved_entities.keys())

        filtered = []
        for match in matches:
            endpoint = match.get("endpoint") or match.get("path", "")
            metadata = match.get("metadata", match)  # fallback to root if inlined
            media_type = SymbolicConstraintFilter._extract_media_type(endpoint)
            consumes = SymbolicConstraintFilter._extract_consumed_entities(metadata)
            supported_intents = SymbolicConstraintFilter._extract_supported_intents(metadata)

            # Media constraint
            if media_pref != "any" and media_type != "any" and media_type != media_pref:
                continue  # mismatch between intent and endpoint media type

            # Entity compatibility
            if not SymbolicConstraintFilter._entities_are_compatible(resolved_keys, consumes):
                continue

            # (Optional) Intent constraint - only apply if intent is strong
            if query_intents:
                required_intent = query_intents[0]
                if not SymbolicConstraintFilter._intent_is_supported(required_intent, supported_intents):
                    continue

            filtered.append(match)

        if not filtered:
            print("⚠️ No symbolic matches passed — falling back to semantic matches")
            return matches

        print(f"🎯 SymbolicConstraintFilter: {len(matches)} → {len(filtered)} after media/entity/intent filtering")
        return filtered

    @staticmethod
    def _infer_media_preference(entities: list) -> str:
        has_tv = "tv" in entities
        has_movie = "movie" in entities
        if has_tv and not has_movie:
            return "tv"
        if has_movie and not has_tv:
            return "movie"
        return "any"

    @staticmethod
    def _extract_media_type(endpoint: str) -> str:
        if "/tv/" in endpoint or "/discover/tv" in endpoint:
            return "tv"
        if "/movie/" in endpoint or "/discover/movie" in endpoint:
            return "movie"
        return "any"

    @staticmethod
    def _extract_consumed_entities(metadata: dict) -> list:
        raw = metadata.get("consumes_entities", "[]")
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except:
            return []

    @staticmethod
    def _extract_supported_intents(metadata: dict) -> list:
        raw = metadata.get("intents", "[]")
        try:
            items = json.loads(raw) if isinstance(raw, str) else raw
            return [item["intent"] for item in items if isinstance(item, dict)]
        except:
            return []

    @staticmethod
    def _entities_are_compatible(resolved_keys: set, consumes_entities: list) -> bool:
        for key in resolved_keys:
            if key in consumes_entities:
                return True
        return False

    @staticmethod
    def _intent_is_supported(intent: str, endpoint_intents: list) -> bool:
        return intent in endpoint_intents
