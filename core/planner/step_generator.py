# core/planner/step_generator.py

from datetime import datetime
from core.entity.param_utils import ParameterMapper
from core.entity.param_utils import resolve_parameter_for_entity


def resolve_path_slots(query_entities=None, entities=None, intents=None):
    entity_types = set()

    if query_entities:
        entity_types = {e["type"]
                        for e in query_entities if isinstance(e, dict)}
    elif entities:
        entity_types = {e.replace("_id", "") for e in entities}

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


def inject_path_slot_parameters(step, resolved_entities, extraction_result=None):
    step.setdefault("parameters", {})
    query_entities = extraction_result.get(
        "query_entities", []) if extraction_result else []
    entities = extraction_result.get(
        "entities", []) if extraction_result else []
    intents = extraction_result.get("intents", []) if extraction_result else []

    path_slots = resolve_path_slots(query_entities, entities, intents)

    for slot, value in path_slots.items():
        if f"{{{slot}}}" in step.get("endpoint", "") and slot not in step["parameters"]:
            step["parameters"][slot] = value

    ParameterMapper.inject_parameters_from_entities(
        query_entities, step["parameters"])
    return step


def inject_parameters_from_query_entities(step, query_entities):
    step.setdefault("parameters", {})
    injected = []

    for ent in query_entities:
        ent_type = ent.get("type")
        resolved_param = resolve_parameter_for_entity(ent_type)
        if not resolved_param or resolved_param in step["parameters"]:
            continue

        value = ent.get("resolved_id") or ent.get("name")
        if not value:
            continue

        step["parameters"][resolved_param] = str(
            value) if isinstance(value, int) else value
        injected.append((resolved_param, value))

    return step


def enrich_plan_with_semantic_parameters(step, query_text, embedding_model, param_collection):
    step.setdefault("parameters", {})
    query_embedding = embedding_model.encode(query_text).tolist()

    results = param_collection.query(
        query_embeddings=[query_embedding], n_results=5)
    inferred_params = results.get("ids", [[]])[0]
    injected = []

    for param in inferred_params:
        if param in step["parameters"]:
            continue
        if param == "vote_average.gte":
            step["parameters"][param] = "7.0"
        elif param == "primary_release_year":
            step["parameters"][param] = str(datetime.now().year)
        elif param != "with_genres":
            step["parameters"][param] = "true"
        injected.append(param)

    return step
