import json

def normalize_parameters(value):
    """
    Normalize TMDB parameters field to ensure it is a dict.

    Handles:
    - JSON strings → parsed dict
    - Lists (e.g. OpenAPI parameter specs) → empty dict
    - Anything else → return only if already dict
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    elif isinstance(value, list):
        return {}
    return value if isinstance(value, dict) else {}

def is_entity_compatible(resolved_keys: set, consumes_entities: list) -> bool:
    """
    Determines whether a given endpoint's required entities are satisfied
    by the resolved entities in the query context.

    - If the endpoint does not consume any entities, it's compatible by default.
    - If it does, at least one resolved key must map to a valid `with_*` param.
    """
    if not consumes_entities:
        return True  # ✅ Allow zero-entity endpoints like /trending/*

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
        param = JOIN_PARAM_MAP.get(key)
        if param and param in consumes_entities:
            return True

    return False

def is_intent_supported(intent: str, endpoint_intents: list) -> bool:
    """
    Return True if the given intent is among the endpoint's declared supported intents.
    """
    return intent in endpoint_intents