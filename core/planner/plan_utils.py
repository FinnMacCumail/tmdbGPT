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
