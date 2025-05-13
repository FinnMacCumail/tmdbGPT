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
        "step": "plan"
    })


def is_symbolically_filterable(endpoint: str) -> bool:
    """
    Returns True if symbolic filtering should apply to this endpoint.
    Only discovery- and search-style endpoints are eligible.
    """
    if not endpoint:
        return False

    # These are the only categories where filtering is meaningful
    filterable_prefixes = (
        "/discover/",
        "/search/",
    )

    # Non-filterable: entity detail endpoints
    excluded_prefixes = (
        "/person/",
        "/company/",
        "/network/",
        "/collection/",
        "/movie/",
        "/tv/",
    )

    # If it's a discovery/search, allow filtering
    for prefix in filterable_prefixes:
        if endpoint.startswith(prefix):
            return True

    # Explicitly reject filtering on detail endpoints
    for prefix in excluded_prefixes:
        if endpoint.startswith(prefix):
            if any(k in endpoint for k in ["/credits", "/tv", "/movie", "/external_ids"]):
                continue  # sub-resource like /credits still may be filterable
            return False

    # Everything else is not filterable by default
    return False
