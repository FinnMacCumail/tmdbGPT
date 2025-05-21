from core.model.evaluator import evaluate_constraint_tree
from core.model.constraint import Constraint
from core.model.constraint import ConstraintGroup


def is_symbol_free_query(state) -> bool:
    """
    Returns True if there are no symbolic query entities (like people, genres, companies),
    and no entity types besides media_type hints like 'tv' or 'movie'.
    """
    extraction = state.extraction_result or {}
    query_entities = extraction.get("query_entities") or []
    entities = extraction.get("entities") or []

    non_symbolic_entities = {"movie", "tv"}

    # Check if any entity besides media hints remains
    real_entities = [e for e in entities if e not in non_symbolic_entities]

    return not query_entities and not real_entities


def route_symbol_free_intent(state):
    """
    Handles intent-only queries without symbolic constraints.
    Returns a plan_steps list with appropriate fallback endpoints.
    """
    intents = state.extraction_result.get("intents", [])
    query = state.input
    media = state.intended_media_type or "all"
    plan_steps = []

    def step(step_id, endpoint, params=None):
        return {
            "step_id": step_id,
            "endpoint": endpoint,
            "method": "GET",
            "parameters": params or {},
            "produces_final_output": True
        }

    # --- TRENDING ---
    if "trending.popular" in intents:
        plan_steps.append(step(
            "step_trending",
            f"/trending/{media}/day"
        ))

    # --- SEARCH ---
    elif "search.multi" in intents:
        plan_steps.append(step(
            "step_search_multi",
            "/search/multi",
            {"query": query}
        ))
    elif "search.person" in intents:
        plan_steps.append(step(
            "step_search_person",
            "/search/person",
            {"query": query}
        ))
    elif "search.movie" in intents:
        plan_steps.append(step(
            "step_search_movie",
            "/search/movie",
            {"query": query}
        ))
    elif "search.tv" in intents:
        plan_steps.append(step(
            "step_search_tv",
            "/search/tv",
            {"query": query}
        ))

    # --- RECOMMENDATIONS ---
    elif "recommendation.similarity" in intents and "movie_id" in state.resolved_entities:
        movie_id = state.resolved_entities["movie_id"][0]
        plan_steps.append(step(
            "step_recommendations",
            f"/movie/{movie_id}/recommendations"
        ))

    # --- DETAILS ---
    elif "details.movie" in intents and "movie_id" in state.resolved_entities:
        movie_id = state.resolved_entities["movie_id"][0]
        plan_steps.append(step(
            "step_details_movie",
            f"/movie/{movie_id}"
        ))
    elif "details.tv" in intents and "tv_id" in state.resolved_entities:
        tv_id = state.resolved_entities["tv_id"][0]
        plan_steps.append(step(
            "step_details_tv",
            f"/tv/{tv_id}"
        ))

    # --- CREDITS ---
    elif "credits.person" in intents and "person_id" in state.resolved_entities:
        pid = state.resolved_entities["person_id"][0]
        plan_steps.append(step(
            "step_person_movie_credits",
            f"/person/{pid}/movie_credits"
        ))
        plan_steps.append(step(
            "step_person_tv_credits",
            f"/person/{pid}/tv_credits"
        ))

    # --- POPULAR / AIRING / UPCOMING LISTS ---
    elif "list.popular" in intents:
        if media == "movie":
            plan_steps.append(step("step_popular_movie", "/movie/popular"))
        elif media == "tv":
            plan_steps.append(step("step_popular_tv", "/tv/popular"))
        else:
            plan_steps.extend([
                step("step_popular_movie", "/movie/popular"),
                step("step_popular_tv", "/tv/popular")
            ])

    elif "list.top_rated" in intents:
        if media == "movie":
            plan_steps.append(step("step_top_rated_movie", "/movie/top_rated"))
        elif media == "tv":
            plan_steps.append(step("step_top_rated_tv", "/tv/top_rated"))
        else:
            plan_steps.extend([
                step("step_top_rated_movie", "/movie/top_rated"),
                step("step_top_rated_tv", "/tv/top_rated")
            ])

    elif "list.upcoming" in intents and media == "movie":
        plan_steps.append(step("step_upcoming_movie", "/movie/upcoming"))

    elif "list.airing_today" in intents and media == "tv":
        plan_steps.append(step("step_airing_today", "/tv/airing_today"))

    elif "list.on_the_air" in intents and media == "tv":
        plan_steps.append(step("step_on_the_air", "/tv/on_the_air"))

    # --- DEFAULT FALLBACK ---
    if not plan_steps:
        plan_steps.append(step(
            "step_generic_fallback",
            f"/trending/{media}/day"
        ))

    return state.model_copy(update={
        "plan_steps": plan_steps,
        "step": "plan",
        "constraint_tree": ConstraintGroup([])  # ✅ Inject empty tree
    })


def is_symbolically_filterable(endpoint: str) -> bool:
    """
    Symbolic filtering should only apply to discovery and search endpoints.
    Do NOT apply symbolic filtering to detail endpoints like /person/{id}.
    """
    return endpoint.startswith("/discover/") or endpoint.startswith("/search/")


def filter_valid_movies_or_tv(entities: list, constraint_tree, registry: dict) -> list:
    """
    Filter a list of TMDB entities (movies or TV) based on symbolic constraint satisfaction.

    Args:
        entities (list): List of TMDB media dicts (must include 'id').
        constraint_tree: ConstraintGroup holding current query constraints.
        registry (dict): Enriched symbolic registry (e.g. state.data_registry).

    Returns:
        list: Entities that pass the symbolic filter.
    """

    if (
        not constraint_tree or
        not getattr(constraint_tree, "constraints", None) or
        len(constraint_tree.constraints) == 0
    ):
        print("✅ Skipping symbolic filtering — no constraints to evaluate.")
        for m in entities:
            m["_provenance"] = m.get("_provenance", {})
            m["_provenance"]["matched_constraints"] = []
            print(
                f"✅ [PASSED] {m.get('title') or m.get('name')} — ID={m.get('id')} — no constraints applied.")
        return entities

    ids = evaluate_constraint_tree(constraint_tree, registry)

    valid_ids = set()
    for media_type in ("movie", "tv"):
        matches = ids.get(media_type, {})
        if not matches:
            continue

        logic = getattr(constraint_tree, "logic", "AND").upper()
        if logic == "OR":
            for match_set in matches.values():
                valid_ids |= match_set
        else:
            intersection = None
            for match_set in matches.values():
                if intersection is None:
                    intersection = set(match_set)
                else:
                    intersection &= match_set
            if intersection:
                valid_ids |= intersection

    filtered = []
    for m in entities:
        mid = m.get("id")
        passed = mid in valid_ids
        m["_provenance"] = m.get("_provenance", {})

        if passed:
            matched = extract_matched_constraints(m, constraint_tree, registry)
            m["_provenance"]["matched_constraints"] = matched
            print(
                f"✅ [PASSED] {m.get('title') or m.get('name')} — ID={mid} — matched: {matched}")
            filtered.append(m)
        else:
            print(
                f"❌ [REJECTED] {m.get('title') or m.get('name')} — ID={mid} — failed symbolic filter")

    return filtered


def extract_matched_constraints(entity, constraint_tree, registry):
    ids = evaluate_constraint_tree(constraint_tree, registry)
    matched = []
    for constraint in constraint_tree:
        if isinstance(constraint, Constraint):
            cid_sets = ids.get(entity.get("media_type", "movie"), {}).get(
                constraint.key, {})
            value = constraint.value
            # ✅ Normalize list-wrapped values like [4495] → 4495
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if value in cid_sets:
                if entity.get("id") in cid_sets[value]:
                    matched.append(f"{constraint.key}={value}")
    return matched
