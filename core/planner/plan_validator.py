import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime

from core.entity.param_utils import ParameterMapper, resolve_parameter_for_entity
from core.llm.focused_endpoints import get_focused_endpoints
from core.model.constraint import Constraint
from core.model.evaluator import evaluate_constraint_tree
from core.planner.question_routing import SymbolicConstraintFilter, QUESTION_TYPE_ROUTING
from core.planner.plan_utils import is_symbolically_filterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
print("✅ plan_validator.py - CHROMA_PATH:", CHROMA_PATH)


class PlanValidator:
    def __init__(self):
        self.client = PersistentClient(path=str(CHROMA_PATH))
        self.param_collection = self.client.get_or_create_collection(
            "tmdb_parameters")
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
                except Exception:
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
            if endpoint_path not in self.PARAM_USED_IN.get(param, []):
                return False
        return True

    def resolve_path_slots(self, query_entities=None, entities=None, intents=None):
        query_entities = query_entities or []
        entity_types = {e["type"]
                        for e in query_entities if isinstance(e, dict)}

        path_params = {}

        if "tv" in entity_types:
            path_params["media_type"] = "tv"
        elif "movie" in entity_types:
            path_params["media_type"] = "movie"
        if "date" in entity_types:
            path_params["time_window"] = "week"

        if intents and "trending.popular" in intents:
            path_params.setdefault("media_type", "movie")
            path_params.setdefault("time_window", "week")

        return path_params

    def inject_path_slot_parameters(self, step, resolved_entities, extraction_result=None):
        step.setdefault("parameters", {})

        query_entities = extraction_result.get(
            "query_entities", []) if extraction_result else []
        entities = extraction_result.get(
            "entities", []) if extraction_result else []
        intents = extraction_result.get(
            "intents", []) if extraction_result else []

        path_slots = self.resolve_path_slots(query_entities, entities, intents)

        for slot, value in path_slots.items():
            if f"{{{slot}}}" in step["endpoint"] and slot not in step["parameters"]:
                step["parameters"][slot] = value

        ParameterMapper.inject_parameters_from_entities(
            query_entities, step["parameters"])
        return step

    def infer_semantic_parameters(self, query_text: str) -> list:
        query_embedding = self.embedding_model.encode(query_text).tolist()
        results = self.param_collection.query(
            query_embeddings=[query_embedding], n_results=5)
        return results.get("ids", [[]])[0]

    def inject_parameters_from_query_entities(self, step: dict, query_entities: list) -> dict:
        step.setdefault("parameters", {})
        for ent in query_entities:
            ent_type = ent.get("type")
            resolved_param = resolve_parameter_for_entity(ent_type)
            if not resolved_param or resolved_param in step["parameters"]:
                continue
            value = ent.get("resolved_id") or ent.get("name")
            if value:
                step["parameters"][resolved_param] = str(value)
        return step

    def validate(self, semantic_matches, state):
        query_entities = state.extraction_result.get("query_entities", [])
        extracted_intents = state.extraction_result.get("intents", [])
        question_type = state.extraction_result.get("question_type")

        required_params = self._resolve_required_parameters_from_entities(
            query_entities)

        filtered_matches = [
            m for m in semantic_matches
            if not required_params or self._endpoint_supports_required_params(m.get("path", ""), required_params)
        ] or semantic_matches

        intent_filtered = []
        for m in filtered_matches:
            supported = m.get("metadata", {}).get("supported_intents", [])
            if not supported or not extracted_intents:
                intent_filtered.append(m)
            elif any(intent in supported for intent in extracted_intents):
                intent_filtered.append(m)
        filtered_matches = intent_filtered or filtered_matches

        for m in filtered_matches:
            if "path" not in m and "endpoint" in m:
                m["path"] = m["endpoint"]

        query = getattr(state, "input", "") or getattr(state, "raw_query", "")
        filtered_matches = apply_llm_endpoint_filter(
            query, filtered_matches, question_type)

        return filtered_matches

    def enrich_plan_with_semantic_parameters(self, step: dict, query_text: str) -> dict:
        step.setdefault("parameters", {})
        inferred_params = self.infer_semantic_parameters(query_text)

        for param in inferred_params:
            if param not in step["parameters"]:
                if param == "vote_average.gte":
                    step["parameters"][param] = "7.0"
                elif param == "primary_release_year":
                    step["parameters"][param] = str(datetime.now().year)
                elif param != "with_genres":
                    step["parameters"][param] = "true"

        return step


# 🔹 External helper function for LLM endpoint filtering
def apply_llm_endpoint_filter(query: str, matches: list, question_type: str) -> list:
    recommended_paths = get_focused_endpoints(
        query, matches, question_type=question_type)
    if not recommended_paths:
        return matches
    return [
        m for m in matches
        if (m.get("path") or m.get("endpoint")) in recommended_paths
    ]


# ✅ Determine whether symbolic filtering (e.g., constraint-based validation) should apply.
# Returns True if:
# - Endpoint is pre-designated as filterable (e.g., /discover/movie), OR
# - Constraint tree includes known symbolic filters (e.g., with_people, with_genres).
def should_apply_symbolic_filter(state, step) -> bool:
    """
    Return True if symbolic constraint filtering should apply to this step.
    Applies if:
    - Endpoint is explicitly symbolically filterable, OR
    - The query includes symbolic constraints like with_people, with_genres, etc.
    """
    if not step:
        return False

    endpoint = step.get("endpoint") or step.get("path") or ""
    if is_symbolically_filterable(endpoint):
        return True

    # 🔍 Fallback override: force filtering if symbolic constraints are present
    constraint_tree = getattr(state, "constraint_tree", None)

    #!!!!!!!!!!!!!!!!!!!!!!!! - this is limited

    if constraint_tree:
        symbolic_keys = {"with_people", "with_genres", "with_keywords",
                         "with_companies", "with_networks", "with_movies",
                         "vote_average.gte", "primary_release_year"}

        for constraint in constraint_tree:
            # 🔍 This is where symbolic constraints (like director_1032) are extracted from the parsed query and incorporated into the constraint tree.
            if isinstance(constraint, Constraint) and constraint.key in symbolic_keys:
                print(f"🔁 Overriding symbolic filter for fallback: {endpoint}")
                return True

    return False
