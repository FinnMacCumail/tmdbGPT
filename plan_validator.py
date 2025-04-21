import json
from chromadb import PersistentClient
from param_utils import ParameterMapper
from llm_client import OpenAILLMClient
from param_utils import is_entity_compatible

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
        "preferred_intents": ["discovery.filtered", "discovery.advanced", "search.movie", "search.person"],
        "fallback_intents": [],
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

    
    def validate(self, semantic_matches, state):
        query_entities = []

        if hasattr(state, "extraction_result"):
            result = getattr(state, "extraction_result")
            if isinstance(result, dict):
                query_entities = result.get("query_entities", [])
            elif hasattr(result, "query_entities"):
                query_entities = result.query_entities

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

        # ✅ Step 2: Normalize .path key so LLM can read it
        for m in filtered_matches:
            if "path" not in m and "endpoint" in m:
                m["path"] = m["endpoint"]
        
        
        #✅ Step 3: Call LLM to get only the relevant endpoints        
        
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

class SymbolicConstraintFilter:   
    INTENT_EQUIVALENTS = {
        # Search-related intents
        "search.multi": [
            "discovery.filtered",
            "credits.person",
            "details.movie",
            "details.tv",
            "search.person",
            "search.movie",
            "search.tv"
        ],
        "search.movie": ["details.movie", "search.multi"],
        "search.tv": ["details.tv", "search.multi"],
        "search.person": ["credits.person", "search.multi"],
        "search.collection": ["collection.details"],

        # Recommendation equivalents
        "recommendation.similarity": ["recommendation"],
        "recommendation.suggested": ["recommendation"],

        # Discovery equivalents
        "discovery.genre_based": ["discovery.filtered"],
        "discovery.temporal": ["discovery.filtered"],
        "discovery.advanced": ["discovery.filtered"],

        # Reviews
        "reviews.movie": ["review.lookup"],
        "reviews.tv": ["review.lookup"],

        # Company/network details
        "companies.studio": ["company.details"],
        "companies.network": ["network.details"],

        # Collections
        "collections.movie": ["collection.details"],

        # Catch-all general fallback
        "search": ["discovery.filtered", "credits.person", "details.movie"],
        "recommendation.similarity": ["recommendation", "discovery.filtered"],
        "recommendation.suggested": ["recommendation", "discovery.filtered"]
    }

    # @staticmethod
    # def apply(matches: list, extraction_result: dict, resolved_entities: dict) -> list:
    #     """
    #     General symbolic filtering based on:
    #     - Entity compatibility (resolved entity must be consumable by endpoint)
    #     - Media type consistency (tv/movie intent should match endpoint media_type)
    #     - Optional: Intent compatibility (e.g., discovery.filtered only matches discover endpoints)
    #     """

    #     question_type = extraction_result.get("question_type")
    #     allowed_intents = set()
    #     if question_type in QUESTION_TYPE_ROUTING:
    #         allowed_intents = set(QUESTION_TYPE_ROUTING[question_type].get("preferred_intents", []))

    #     entities = extraction_result.get("entities", [])
    #     query_intents = extraction_result.get("intents", [])
    #     media_pref = SymbolicConstraintFilter._infer_media_preference(entities)
    #     resolved_keys = set(resolved_entities.keys())
    #     print("\n🔎 Symbolic Filter Debug — Candidates:")
    #     for m in matches:
    #         print(f"• {m.get('endpoint')}")
    #         print(f"  → media_type: {SymbolicConstraintFilter._extract_media_type(m.get('endpoint', ''))}")
    #         print(f"  → consumes_entities: {SymbolicConstraintFilter._extract_consumed_entities(m)}")
    #         print(f"  → supported_intents: {SymbolicConstraintFilter._extract_supported_intents(m)}")
    #     filtered = []
    #     for match in matches:
    #         endpoint = match.get("endpoint") or match.get("path", "")
    #         metadata = match.get("metadata", match)  # fallback to root if inlined
    #         media_type = SymbolicConstraintFilter._extract_media_type(endpoint)
    #         consumes = SymbolicConstraintFilter._extract_consumed_entities(metadata)
    #         supported_intents = SymbolicConstraintFilter._extract_supported_intents(metadata)
    #         final_score = match.get("final_score", 0)

    #         media_ok = (media_pref == "any" or media_type == "any" or media_type == media_pref)
    #         # Symbolic entity filtering: always allow if no entities are required
    #         entities_ok = is_entity_compatible(resolved_keys, consumes)
    #         if not consumes:
    #             print(f"  ⚠️ No entity required — allowing endpoint through with matching intent only")

    #         # Intent match with fallback equivalence logic
    #         intent = query_intents[0] if query_intents else None
    #         intent_ok = SymbolicConstraintFilter._intent_is_supported(
    #             intent, supported_intents, question_type
    #         )
            
    #         intent_overlap = bool(set(supported_intents) & allowed_intents)

    #         print(f"\n• {endpoint}")
    #         print(f"  🔹 score: {final_score}")
    #         print(f"  🔹 media_type: {media_type} (query: {media_pref}) → {'✅' if media_ok else '❌'}")
    #         print(f"  🔹 consumes_entities: {consumes} (resolved: {list(resolved_keys)}) → {'✅' if entities_ok else '❌'}")
    #         print(f"  🔹 supported_intents: {supported_intents} (query: {query_intents}) → {'✅' if intent_ok else '❌'}")

    #         print(f"  🔹 allowed_intents (for type={question_type}): {allowed_intents}")
    #         print(f"  🔹 intent_overlap: {intent_overlap}")

    #         if media_ok and entities_ok and intent_ok and intent_overlap:
    #             print("  ✅ INCLUDED in symbolic matches")
    #             filtered.append(match)
    #         else:
    #             print("  ❌ EXCLUDED from symbolic matches")
    #     return filtered
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

            # NEW INTENT FALLBACK CHECK
            intent_overlap = any(intent in allowed_intents for intent in supported_intents)

            if intent_overlap:
                print(f"✅ Allowed intent overlap: {supported_intents} matches allowed intents {allowed_intents}")
                filtered.append(match)
            else:
                print(f"❌ Excluded endpoint '{endpoint}' due to intent mismatch (supported: {supported_intents}, allowed: {allowed_intents})")

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
        """
        Map resolved entity keys like 'company_id' to 'with_companies'
        and compare against what the endpoint actually supports.
        """
        JOIN_PARAM_MAP = {
            "person_id": "with_people",
            "genre_id": "with_genres",
            "company_id": "with_companies",
            "network_id": "with_networks",
            "collection_id": "with_collections",
            "keyword_id": "with_keywords",
            "tv_id": "with_tv",
            "movie_id": "with_movies"
        }

        for key in resolved_keys:
            mapped_param = JOIN_PARAM_MAP.get(key)
            if mapped_param and mapped_param in consumes_entities:
                return True

        return False

    @staticmethod
    def _intent_is_supported(intent: str, endpoint_intents: list, question_type: str) -> bool:
        if not intent or not endpoint_intents:
            return False

        # Fetch preferred and fallback intents from routing matrix
        routing = QUESTION_TYPE_ROUTING.get(question_type, {})
        allowed_intents = set(routing.get("preferred_intents", []) + routing.get("fallback_intents", []))

        # Direct match or fallback equivalence
        if intent in allowed_intents and set(endpoint_intents) & allowed_intents:
            return True

        # Apply symbolic intent equivalence from INTENT_EQUIVALENTS
        for alias in SymbolicConstraintFilter.INTENT_EQUIVALENTS.get(intent, []):
            if alias in endpoint_intents:
                print(f"🔁 Intent fallback matched: {intent} → {alias}")
                return True

        return False


