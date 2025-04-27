import json
from chromadb import PersistentClient
from param_utils import ParameterMapper
from llm_client import OpenAILLMClient
from sentence_transformers import SentenceTransformer

QUESTION_TYPE_ROUTING = {
    "count": {
        "preferred_intents": ["credits.person", "credits.movie", "credits.tv"],
        "fallback_intents": ["details.person"],
        "response_format": "count_summary",
        "description": "Returns a numeric count of appearances"
    },
    "summary": {
        "preferred_intents": ["details.person", "details.movie", "details.tv"],
        "fallback_intents": [],
        "response_format": "summary",
        "description": "Returns a biography or synopsis"
    },
    "fact": {
        "preferred_intents": ["details.movie", "details.person"],
        "fallback_intents": [],
        "response_format": "summary",
        "description": "Provides a factual answer about a movie or person"
    },
    "timeline": {
        "preferred_intents": ["credits.person", "credits.tv", "credits.movie"],
        "fallback_intents": ["details.movie"],
        "response_format": "timeline",
        "description": "Returns entries ordered by release date"
    },
    "comparison": {
        "preferred_intents": ["details.movie", "details.person"],
        "fallback_intents": [],
        "response_format": "comparison",
        "description": "Returns a side-by-side comparison"
    },
    "list": {
        "preferred_intents": ["discovery.filtered", "discovery.advanced", "search.movie", "search.person", "credits.person"],
        "fallback_intents": [],
        "allowed_wildcards": ["discovery."],
        "response_format": "list",
        "description": "Returns a list of matching items"
    }
}



class PlanValidator:
    def __init__(self):
        self.client = PersistentClient(path="./chroma_db")
        self.param_collection = self.client.get_or_create_collection("tmdb_parameters")
        self.PARAM_USED_IN = {}
        self._preload_parameter_usage()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
            "rating": "vote_average.gte",  
            "date": "primary_release_year" 
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
        if "parameters" not in step:
            step["parameters"] = {}

        # ✅ Extract components from the extraction result
        query_entities = extraction_result.get("query_entities", []) if extraction_result else []
        entities = extraction_result.get("entities", []) if extraction_result else []
        intents = extraction_result.get("intents", []) if extraction_result else []

        # ✅ Resolve implicit path parameters (e.g., media_type, time_window)
        path_slots = self.resolve_path_slots(
            query_entities=query_entities,
            entities=entities,
            intents=intents
        )

        for slot, value in path_slots.items():
            if f"{{{slot}}}" in step["endpoint"] and slot not in step["parameters"]:
                step["parameters"][slot] = value
                print(f"🧩 Injected path slot: {slot} = {value} into {step['endpoint']}")

        # ✅ Inject value-based query filters (e.g. rating, year) using mapped param logic
        ParameterMapper.inject_parameters_from_entities(query_entities, step["parameters"])

        print(f"✅ Final injected parameters: {step.get('parameters', {})}")
        return step

    # phase 2.2 - Semantic Parameter Inference (Dynamic Enhancement)
    def infer_semantic_parameters(self, query_text: str) -> list:
        """
        Perform semantic search over TMDB parameters to suggest useful filters
        based on the meaning of the user's query.

        Args:
            query_text (str): Natural language query from user.

        Returns:
            list: List of suggested TMDB parameter names.
        """
        # Embed the query locally
        query_embedding = self.embedding_model.encode(query_text).tolist()

        # Search the TMDB parameters collection
        results = self.param_collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        # Extract suggested parameter names
        suggested_parameters = results.get("ids", [[]])[0]
        return suggested_parameters
    
    def inject_parameters_from_query_entities(self, step: dict, query_entities: list) -> dict:
        """
        Dynamically inject missing parameters into a plan step based on extracted query entities.
        Example: If the user mentions Crime genre, inject with_genres automatically.
        """
        from param_utils import resolve_parameter_for_entity

        if "parameters" not in step:
            step["parameters"] = {}

        injected = []

        for ent in query_entities:
            ent_type = ent.get("type")
            resolved_param = resolve_parameter_for_entity(ent_type)
            if not resolved_param:
                continue

            # Avoid overwriting if already planned
            if resolved_param in step["parameters"]:
                continue

            # Inject based on entity name or resolved ID
            value = ent.get("resolved_id") or ent.get("name")
            if not value:
                continue

            if isinstance(value, int):
                step["parameters"][resolved_param] = str(value)
            else:
                step["parameters"][resolved_param] = value

            injected.append((resolved_param, value))

        if injected:
            print(f"✅ Injected parameters from query entities: {injected}")

        return step

    def validate(self, semantic_matches, state):
        query_entities = []
        extracted_intents = []

        if hasattr(state, "extraction_result"):
            result = getattr(state, "extraction_result")
            if isinstance(result, dict):
                query_entities = result.get("query_entities", [])
                extracted_intents = result.get("intents", [])
            elif hasattr(result, "query_entities"):
                query_entities = result.query_entities
                extracted_intents = getattr(result, "intents", [])

        else:
            extracted_intents = []

        # ✅ Step 1: Ensure param compatibility was handled
        required_params = self._resolve_required_parameters_from_entities(query_entities)
        print(f"🔍 Resolved required parameters: {required_params}")

        if not required_params:
            print("🔎 No symbolic filtering required — using all semantic matches.")
            filtered_matches = semantic_matches
        else:
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
                filtered_matches = semantic_matches

        # ✅ Step 1.5: Ensure intent compatibility
        print(f"🔍 Checking intent compatibility: {extracted_intents}")
        intent_filtered_matches = []
        for m in filtered_matches:
            supported_intents = m.get("metadata", {}).get("supported_intents", [])
            if not supported_intents or not extracted_intents:
                # No intent constraint, allow
                intent_filtered_matches.append(m)
            elif any(intent in supported_intents for intent in extracted_intents):
                intent_filtered_matches.append(m)
            else:
                print(f"❌ Excluded endpoint '{m['path']}' due to intent mismatch (supported: {supported_intents}, allowed: {extracted_intents})")

        filtered_matches = intent_filtered_matches

        if not filtered_matches:
            print("⚠️ No endpoints matched extracted intents, falling back to semantic matches.")
            filtered_matches = semantic_matches

        # ✅ Step 2: Normalize .path key so LLM can read it
        for m in filtered_matches:
            if "path" not in m and "endpoint" in m:
                m["path"] = m["endpoint"]

        # ✅ Step 3: Call LLM to get only the relevant endpoints
        query = getattr(state, "input", "") or getattr(state, "raw_query", "")
        question_type = state.extraction_result.get("question_type")
        llm = OpenAILLMClient()
        recommended = llm.get_focused_endpoints(query, filtered_matches, question_type=question_type)
        print(f"📤 LLM recommended endpoints: {recommended}")
        if recommended:
            before = len(filtered_matches)
            filtered_matches = [
                m for m in filtered_matches
                if m.get("path") in recommended or m.get("endpoint") in recommended
            ]
            print(f"🧭 LLM endpoint focus pruning: {before} → {len(filtered_matches)}")

        return filtered_matches
    
    def enrich_plan_with_semantic_parameters(self, step: dict, query_text: str) -> dict:
        """
        After basic injection, perform semantic inference to suggest missing parameters 
        based on the meaning of the query.
        """
        if "parameters" not in step:
            step["parameters"] = {}

        inferred_params = self.infer_semantic_parameters(query_text)
        injected = []

        for param in inferred_params:
            # Only inject if not already present
            if param not in step["parameters"]:
                # Handle special cases for common semantic params
                if param == "vote_average.gte":
                    step["parameters"][param] = "7.0"  # Assume top-rated means 7+
                    injected.append((param, "7.0"))
                elif param == "primary_release_year":
                    from datetime import datetime
                    current_year = datetime.now().year
                    step["parameters"][param] = str(current_year)
                    injected.append((param, str(current_year)))
                elif param == "with_genres":
                    # Don't inject arbitrary genres here — only if genre mentioned
                    continue
                else:
                    # Default fallback
                    step["parameters"][param] = "true"
                    injected.append((param, "true"))

        if injected:
            print(f"✨ Semantically inferred parameters injected: {injected}")

        return step


class SymbolicConstraintFilter:
    MEDIA_FILTER_ENTITIES = {"genre", "rating", "date", "runtime", "votes", "language", "country"}

    @staticmethod
    def is_media_endpoint(produces_entities: list) -> bool:
        return any(media in produces_entities for media in {"movie", "tv"})

    @staticmethod
    def apply(matches: list, extraction_result: dict, resolved_entities: dict) -> list:
        question_type = extraction_result.get("question_type")
        query_intents = extraction_result.get("intents", [])
        resolved_keys = set(resolved_entities.keys())

        routing = QUESTION_TYPE_ROUTING.get(question_type, {})
        allowed_intents = set(routing.get("preferred_intents", []) + routing.get("fallback_intents", []))

        filtered = []

        for match in matches:
            endpoint = match.get("endpoint") or match.get("path", "")
            metadata = match.get("metadata", match)
            supported_intents = SymbolicConstraintFilter._extract_supported_intents(metadata)
            consumes = SymbolicConstraintFilter._extract_consumed_entities(metadata)
            produces = SymbolicConstraintFilter._extract_produced_entities(metadata)

            # --- Entity Penalty ---
            missing_required_entities = []
            for key in resolved_keys:
                if key == "__query":
                    continue  # ✅ Ignore query text entity

                entity_type = SymbolicConstraintFilter._map_key_to_entity(key)
                if entity_type not in consumes:
                    if entity_type in SymbolicConstraintFilter.MEDIA_FILTER_ENTITIES:
                        if SymbolicConstraintFilter.is_media_endpoint(produces):
                            print(f"⚡ Skipping penalty for {entity_type} because endpoint produces media items.")
                            continue
                    missing_required_entities.append(key)

            # --- Soft Relaxation ---
            soft_relaxed_fields = []
            if missing_required_entities:
                for key in missing_required_entities:
                    entity_type = SymbolicConstraintFilter._map_key_to_entity(key)
                    if entity_type in SymbolicConstraintFilter.MEDIA_FILTER_ENTITIES:
                        soft_relaxed_fields.append(key)

                if soft_relaxed_fields and len(soft_relaxed_fields) == len(missing_required_entities):
                    print(f"⚡ Soft relaxing missing filters: {soft_relaxed_fields}")
                    match["soft_relaxed"] = soft_relaxed_fields
                    match["force_allow"] = True  # ✅ NEW: force allow intent match
                elif "credits" in endpoint and SymbolicConstraintFilter.is_media_endpoint(produces):
                    print(f"⚡ Force-allowing {endpoint} because it produces media items via credits endpoint.")
                    match["force_allow"] = True  # ✅ NEW: allow credits endpoints

            entity_penalty = 0.0
            if missing_required_entities:
                if SymbolicConstraintFilter.is_media_endpoint(produces):
                    print(f"⚡ Allowing media-producing endpoint {endpoint} even though missing: {missing_required_entities}")
                    entity_penalty = 0.0
                else:
                    entity_penalty = 0.2 * len(missing_required_entities)
                    print(f"⚠️ Entity penalty on '{endpoint}' for missing: {missing_required_entities}")

            # --- Question Type Penalty ---
            qt_penalty = 0.0
            if question_type:
                expected_patterns = SymbolicConstraintFilter._expected_patterns_for_question_type(question_type)
                endpoint_lower = endpoint.lower()
                if expected_patterns and not any(pat in endpoint_lower for pat in expected_patterns):
                    if SymbolicConstraintFilter.is_media_endpoint(produces):
                        print(f"⚡ Skipping question type penalty because endpoint produces media items.")
                    else:
                        qt_penalty = 0.3
                        print(f"⚠️ Question type mismatch on '{endpoint}' for question_type='{question_type}' (expected {expected_patterns})")

            # --- Apply Penalties ---
            existing_penalty = match.get("penalty", 0.0)
            total_penalty = entity_penalty + qt_penalty
            match["penalty"] = existing_penalty + total_penalty

            # --- Intent Overlap or Soft Relaxed Check ---
            intent_overlap = any(intent in allowed_intents for intent in supported_intents)

            # 🧠 NEW: Dynamic fallback matching
            routing = QUESTION_TYPE_ROUTING.get(question_type, {})
            wildcards = routing.get("allowed_wildcards", [])

            if not intent_overlap and wildcards:
                for wildcard in wildcards:
                    if any(supported_intent.startswith(wildcard) for supported_intent in supported_intents):
                        print(f"⚡ Allowed wildcard intent '{supported_intents}' because matches wildcard '{wildcard}' for question_type '{question_type}'.")
                        intent_overlap = True
                        break

            if intent_overlap:
                print(f"✅ Allowed intent overlap: {supported_intents} matches allowed intents {allowed_intents}")
                filtered.append(match)
            else:
                if match.get("soft_relaxed") or match.get("force_allow"):
                    print(f"⚡ Allowing endpoint {endpoint} because soft-relaxed or force-allowed")
                    filtered.append(match)
                else:
                    print(f"❌ Excluded endpoint '{endpoint}' due to intent mismatch (supported: {supported_intents}, allowed: {allowed_intents})")

        return filtered
    
    @staticmethod
    def _map_key_to_entity(key: str) -> str:
        if key.endswith("_id"):
            return key[:-3]  # person_id → person
        return key
    

    @staticmethod
    def _extract_consumed_entities(metadata: dict) -> list:
        raw = metadata.get("consumes_entities", "[]")
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except:
            return []

    @staticmethod
    def _extract_produced_entities(metadata: dict) -> list:
        raw = metadata.get("produces_entities", "[]")
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except:
            return []

    
    @staticmethod
    def _extract_supported_intents(metadata: dict) -> list:
        raw = metadata.get("supported_intents", [])
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return []
        return raw

    @staticmethod
    def _expected_patterns_for_question_type(question_type: str) -> list:
        # Rough mappings based on QUESTION_TYPE_ROUTING expectations
        mapping = {
            "count": ["/credits", "/details", "/movie_credits", "/tv_credits"],
            "summary": ["/details", "/person", "/movie", "/tv"],
            "timeline": ["/credits", "/discover"],
            "comparison": ["/details", "/credits"],
            "list": ["/discover", "/search", "/person", "/credits"],
            "fact": ["/details", "/movie", "/person"]
        }
        return mapping.get(question_type, [])
    
    @staticmethod
    def prioritize_media_type(matches, extraction_result):
        """
        Boost endpoint matches where media_type aligns with user intent.
        """
        intents = extraction_result.get("intents", [])

        if any("tv" in intent.lower() for intent in intents):
            desired_media_type = "tv"
        elif any("movie" in intent.lower() for intent in intents):
            desired_media_type = "movie"
        else:
            desired_media_type = None

        if not desired_media_type:
            return matches  # No strong signal, leave matches as is

        for match in matches:
            media_type = match.get("metadata", {}).get("media_type")
            if media_type == desired_media_type:
                match["score"] += 0.2  # Boost correct media type
            elif media_type:
                match["score"] -= 0.2  # Penalize wrong media type

        return matches        

