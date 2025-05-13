import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb
from nlp.llm_client import OpenAILLMClient
from core.planner.entity_reranker import EntityAwareReranker
from core.entity.param_utils import normalize_parameters
from core.planner.plan_validator import PlanValidator
import logging
from core.entity.param_utils import resolve_parameter_for_entity
from core.llm.extractor import extract_entities_and_intents

load_dotenv()

# Suppress SentenceTransformer logs before instantiation
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# 📦 Path-safe project root reference
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Init clients
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = chroma_client.get_or_create_collection("tmdb_endpoints")
openai_client = OpenAILLMClient()


class MediaTypeAwareReranker:
    @staticmethod
    def boost_by_media_type(matches: list, extraction_result: dict, boost_weight: float = 0.2) -> list:
        """
        Boost final_score if endpoint's media_type matches query's intended media_type (tv/movie).
        """
        intents = extraction_result.get("intents", [])

        # Infer desired media type
        if any("tv" in intent.lower() for intent in intents):
            desired_media_type = "tv"
        elif any("movie" in intent.lower() for intent in intents):
            desired_media_type = "movie"
        else:
            desired_media_type = None

        if not desired_media_type:
            return matches  # No strong signal

        for m in matches:
            try:
                media_type = m.get("media_type") or m.get(
                    "metadata", {}).get("media_type", "any")
            except Exception:
                media_type = "any"

            if media_type == desired_media_type:
                m["final_score"] += boost_weight
                m["final_score"] = round(m["final_score"], 3)
            elif media_type != "any":
                m["final_score"] -= boost_weight
                m["final_score"] = round(m["final_score"], 3)

        return sorted(matches, key=lambda x: x["final_score"], reverse=True)


class ParameterAwareReranker:
    @staticmethod
    def boost_by_supported_parameters(matches: list, query_entities: list, boost_weight: float = 0.1) -> list:
        """
        Boost final_score if endpoint supports parameters matching query entity types.
        """

        for m in matches:
            try:
                supports_parameters = json.loads(
                    m.get("supports_parameters", "[]"))
            except Exception:
                supports_parameters = []

            match_count = 0

            for ent in query_entities:
                entity_type = ent.get("type")
                if not entity_type:
                    continue

                resolved_param = resolve_parameter_for_entity(entity_type)
                if resolved_param and resolved_param in supports_parameters:
                    match_count += 1

            if match_count:
                m["final_score"] += match_count * boost_weight
                m["final_score"] = round(m["final_score"], 3)

        return sorted(matches, key=lambda x: x["final_score"], reverse=True)


# Map entity types to their join query parameters
JOIN_PARAM_MAP = {
    "person_id": "with_people",
    "genre_id": "with_genres",
    "company_id": "with_companies",
    "keyword_id": "with_keywords",
    "network_id": "with_networks",
    "collection_id": "with_collections",
    "tv_id": "with_tv",
    "movie_id": "with_movies"
}


def retrieve_semantic_matches(prompt: str, top_k: int = 10) -> list:
    """
    Interpret a natural language prompt to generate structured extraction,
    then perform hybrid semantic retrieval.

    Args:
        prompt (str): user-style query like "Fetch endpoints requiring person_id for intent credits.person"
        top_k (int): number of results to return

    Returns:
        list: retrieved endpoint matches
    """
    structured = extract_entities_and_intents(prompt)
    return rank_and_score_matches(structured, top_k=top_k)


def compute_match_score(user_extraction, candidate_metadata):
    user_intents = set(user_extraction.get("intents", []))
    user_entities = set(user_extraction.get("entities", []))
    query_entities = [e for e in user_extraction.get(
        "query_entities", []) if isinstance(e, dict)]

    try:
        endpoint_intents = json.loads(candidate_metadata.get("intents", "[]"))
    except json.JSONDecodeError:
        endpoint_intents = []

    if isinstance(endpoint_intents, dict):
        endpoint_intents = [endpoint_intents]

    endpoint_entities = set(candidate_metadata.get("entities", "").split(", "))
    path = candidate_metadata.get("path", "")

    # --- Normalized intent score
    matched_intents = [ei for ei in endpoint_intents if ei.get(
        "intent") in user_intents]
    intent_score = sum(float(i.get("confidence", 0)) for i in matched_intents)
    if matched_intents:
        intent_score /= len(matched_intents)
    intent_score = min(intent_score, 1.0)

    # --- Entity score
    weights = {"movie": 0.5, "year": 0.3, "genre": 0.4,
               "rating": 0.4, "person": 0.4, "date": 0.3}
    if "discovery.filtered" in user_intents:
        weights.update({"rating": 0.6, "year": 0.5})
    elif "trending.popular" in user_intents:
        weights.update({"year": 0.1, "rating": 0.1})

    entity_score = sum(weights.get(e, 0.1)
                       for e in user_entities & endpoint_entities)

    # --- Param compatibility
    try:
        endpoint_params = json.loads(
            candidate_metadata.get("parameters", "[]"))
        param_names = {p.get("name", "")
                       for p in endpoint_params if isinstance(p, dict)}
    except Exception:
        param_names = set()
    param_overlap = len([e for e in user_entities if e in param_names])
    param_boost = param_overlap / max(1, len(user_entities))

    # --- Principled synergy boost: multi-person queries + with_people
    query_persons = [e for e in query_entities if e.get("type") == "person"]
    if "with_people" in param_names and len(query_persons) >= 2:
        entity_score += 0.6

    # --- Penalties
    generic = {"keyword", "date", "rating", "country"}
    unrelated_entities = endpoint_entities - user_entities - generic
    mismatch_penalty = 0.15 * len(unrelated_entities)

    if "movie" in user_entities and "tv" in endpoint_entities:
        mismatch_penalty += 0.15
    if any(i.startswith("trending") for i in user_intents) and "/search" in path:
        mismatch_penalty += 0.2
    if "general" in [i.get("intent") for i in endpoint_intents]:
        mismatch_penalty += 0.2

    score = intent_score + entity_score + 0.5 * param_boost - mismatch_penalty
    return round(score, 3)


def rank_and_score_matches(extraction_result, top_k=10):
    embedding_text = json.dumps(extraction_result)
    query_embedding = embedder.encode(embedding_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    matches = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        # ✅ Normalize parameters (crucial fix)
        metadata["parameters"] = normalize_parameters(
            metadata.get("parameters", {}))

        score = compute_match_score(extraction_result, metadata)
        final_score = 0.7 * score + 0.3 * (1 - distance)

        metadata.update({
            "endpoint": metadata.get("path"),
            "match_score": round(score, 3),
            "distance": round(distance, 4),
            "final_score": round(final_score, 3),
            "score": round(final_score, 3),  # for reranker compatibility
        })

        matches.append(metadata)

    # 🧠 Boost by matched named entities (if any)
    matches = EntityAwareReranker.boost_by_entity_mentions(
        matches, extraction_result.get("query_entities", [])
    )

    matches = ParameterAwareReranker.boost_by_supported_parameters(
        matches, extraction_result.get("query_entities", [])
    )

    matches = MediaTypeAwareReranker.boost_by_media_type(
        matches, extraction_result)

    return matches


def convert_matches_to_execution_steps(matches, extraction_result, resolved_entities):
    """
    Convert hybrid search matches into executable step dictionaries.
    Normalizes parameters and injects resolved values.
    """
    steps = []
    query_entity = None

    query_entities = extraction_result.get("query_entities", [])
    if query_entities and isinstance(query_entities[0], dict):
        query_entity = query_entities[0].get("name")

    for idx, match in enumerate(matches):
        endpoint = match.get("endpoint") or match.get("path")
        method = match.get("method", "GET")
        raw_params = match.get("parameters", {})
        parameters = normalize_parameters(raw_params)

        if not isinstance(parameters, dict):
            # print(f"❌ Parameter normalization failed for {endpoint}: type={type(parameters)} → forcing empty dict")
            parameters = {}
        else:
            assert isinstance(
                parameters, dict), f"🔴 Parameters not a dict after normalization: {endpoint}"

        # 🔁 Inject resolved path-style entity_id substitutions
        for entity_key, entity_value in resolved_entities.items():
            if f"{{{entity_key}}}" in endpoint:
                parameters[entity_key] = entity_value[0] if isinstance(
                    entity_value, list) else entity_value

        # 🧩 Inject resolved IDs for with_* joins
        for entity_key, param_name in JOIN_PARAM_MAP.items():
            if entity_key in resolved_entities:
                ids = resolved_entities[entity_key]
                if isinstance(ids, list):
                    parameters[param_name] = ",".join(map(str, ids))
                else:
                    parameters[param_name] = str(ids)

        # 🔍 Inject LLM query_entity as search string
        if "/search/" in endpoint and "query" not in parameters and query_entity:
            parameters["query"] = query_entity

        step = {
            "step_id": f"step_{idx}",
            "endpoint": endpoint,
            "method": method,
            "parameters": parameters
        }

        # tmdb up -
        plan_validator = PlanValidator()
        step = plan_validator.inject_parameters_from_query_entities(
            step, extraction_result.get("query_entities", []))
        # 🔥 Then: enrich with semantic parameters (Phase 18.1 starter)
        query_text = extraction_result.get(
            "query_text") or extraction_result.get("raw_query") or ""
        if query_text:
            step = plan_validator.enrich_plan_with_semantic_parameters(
                step, query_text=query_text
            )

        steps.append(step)

    return steps
