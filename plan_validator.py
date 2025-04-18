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

    def resolve_path_slots(self, query_entities=None, entities=None):
        query_entities = query_entities or []
        entity_types = set()

        if query_entities:
            entity_types = {e["type"] for e in query_entities if isinstance(e, dict)}
        elif entities:
            entity_types = set(entities)

        PATH_PARAM_SLOT_MAP = {
            "movie": "media_type",
            "tv": "media_type",
            "date": "time_window"
        }

        path_params = {}
        for ent_type in entity_types:
            slot_name = PATH_PARAM_SLOT_MAP.get(ent_type)
            if slot_name == "media_type" and ent_type in ["movie", "tv"]:
                path_params[slot_name] = ent_type
            elif slot_name == "time_window" and ent_type == "date":
                path_params[slot_name] = "week"
        return path_params

    def inject_path_slot_parameters(self, step, state):
        query_entities = []

        if hasattr(state, "extraction_result"):
            result = getattr(state, "extraction_result")
            if isinstance(result, dict):
                query_entities = result.get("query_entities", [])
            elif hasattr(result, "query_entities"):
                query_entities = result.query_entities

        entities = getattr(state, "entities", []) if hasattr(state, "entities") else []
        path_slots = self.resolve_path_slots(query_entities=query_entities, entities=entities)

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
