from typing import List, Dict

RESPONSE_RENDERERS = {}

def register_renderer(name):
    def decorator(func):
        RESPONSE_RENDERERS[name] = func
        return func
    return decorator

class ResponseFormatter:
    @staticmethod
    def format_responses(responses: List[dict]) -> List[str]:
        def tag_for_score(score: float) -> str:
            if score >= 1.0:
                return "🎯"
            elif score >= 0.75:
                return "✅"
            elif score >= 0.5:
                return "➖"
            else:
                return "⚠️"

        def tag_for_source(source: str) -> str:
            return "♻️" if "_relaxed_" in source else ""

        enriched = []
        for r in responses:
            if isinstance(r, str):
                # 🎯 NEW: Defensive handling for plain strings
                enriched.append({"line": r, "score": 0.3})
                continue

            if isinstance(r, dict) and r.get("type") == "movie_summary":
                title = r.get("title", "Untitled")
                overview = r.get("overview", "")
                score = r.get("final_score", 1.0)
                source = r.get("source", "")
                tag = tag_for_score(score)
                fallback_tag = tag_for_source(source)
                badge = ""
                if "cast" in overview.lower():
                    badge += " 🎭"
                if "directed" in overview.lower() or "director" in overview.lower():
                    badge += " 🎬"
                line = f"{tag} {title}: {overview}{badge}"
                if source:
                    line += f"  (from {source})"
                if fallback_tag:
                    line = fallback_tag + " " + line
                enriched.append({"line": line, "score": score})
            else:
                enriched.append({"line": f"📎 {str(r)}", "score": 0.2})

        sorted_lines = sorted(enriched, key=lambda x: x["score"], reverse=True)
        return [entry["line"] for entry in sorted_lines]

@register_renderer("count_summary")
def format_count_summary(state) -> dict:
    movie_count = 0
    tv_count = 0
    for r in state.responses:
        if r.get("type") == "movie_summary":
            src = r.get("source", "")
            if "movie_credits" in src:
                movie_count += 1
            elif "tv_credits" in src:
                tv_count += 1

    name = state.extraction_result.get("query_entities", [{}])[0].get("name", "This person")
    text = f"🎬 {name} has appeared in {movie_count} movie(s) and {tv_count} TV show(s)."
    return {
        "response_format": "count_summary",
        "question_type": "count",
        "entity": name,
        "text": text,
        "entries": [text]
    }

@register_renderer("ranked_list")
def format_ranked_list(state) -> dict:
    lines = ResponseFormatter.format_responses(state.responses)
    return {
        "response_format": "ranked_list",
        "question_type": "list",
        "entries": lines[:10]
    }

@register_renderer("summary")
def format_summary(state) -> dict:
    responses = state.responses
    SUMMARY_TYPES = {
        "movie_summary": "🎬",
        "tv_summary": "📺",
        "person_profile": "👤",
        "company_profile": "🏢",
        "network_profile": "📡",
        "collection_profile": "🎞️",
    }

    for r in responses:
        r_type = r.get("type")
        emoji = SUMMARY_TYPES.get(r_type, "📄")
        title = r.get("title", "Untitled")
        overview = r.get("overview", "No summary available.")
        text = f"{emoji} {title}: {overview}"

        return {
            "response_format": "summary",
            "question_type": "summary",
            "entity": title,
            "text": text,
            "entries": [text]
        }

    # Fallback if no supported response found
    from response_formatter import generate_explanation
    explanation = generate_explanation(state.extraction_result)
    return {
        "response_format": "summary",
        "question_type": "summary",
        "text": f"⚠️ {explanation or 'No summary available.'}",
        "entries": [f"⚠️ {explanation or 'No summary available.'}"]
    }


@register_renderer("timeline")
def format_timeline(state) -> dict:
    import re
    entries = []
    for item in state.responses:
        if item.get("type") != "movie_summary":
            continue
        title = item.get("title", "Untitled")
        overview = item.get("overview", "No synopsis available.")
        source = item.get("source", "")
        score = item.get("final_score", 1.0)
        year = item.get("release_date", "")[:4] if "release_date" in item else None
        if not year:
            match = re.search(r"(19|20)\d{2}", overview)
            year = match.group(0) if match else None
        entries.append({
            "title": title,
            "overview": overview,
            "release_year": int(year) if year and year.isdigit() else None,
            "source": source,
            "score": score
        })
    entries.sort(key=lambda x: x.get("release_year") or 3000)
    name = state.extraction_result.get("query_entities", [{}])[0].get("name", "")
    return {
        "response_format": "timeline",
        "question_type": "timeline",
        "entity": name,
        "entries": entries
    }

@register_renderer("comparison")
def format_comparison(state) -> dict:
    query_entities = state.extraction_result.get("query_entities", [])
    if len(query_entities) != 2:
        return {"response_format": "comparison", "question_type": "comparison", "error": "Comparison requires exactly two entities."}

    left_id = str(query_entities[0].get("resolved_id"))
    right_id = str(query_entities[1].get("resolved_id"))
    left_name = query_entities[0].get("name", "Entity A")
    right_name = query_entities[1].get("name", "Entity B")

    left_entries = []
    right_entries = []

    for r in state.responses:
        if r.get("type") != "movie_summary":
            continue
        src = r.get("source", "")
        entry = {
            "title": r.get("title", ""),
            "overview": r.get("overview", ""),
            "score": r.get("final_score", 1.0),
            "source": src
        }
        if left_id in src:
            left_entries.append(entry)
        elif right_id in src:
            right_entries.append(entry)

    left_entries.sort(key=lambda x: x["score"], reverse=True)
    right_entries.sort(key=lambda x: x["score"], reverse=True)

    return {
        "response_format": "comparison",
        "question_type": "comparison",
        "left": {"name": left_name, "entries": left_entries[:3]},
        "right": {"name": right_name, "entries": right_entries[:3]}
    }

def generate_explanation(extraction_result: dict) -> str:
    """
    Create a human-readable explanation of what the search tried to do.

    Args:
        extraction_result (dict): Output from extract_entities_and_intents()

    Returns:
        str: Natural language explanation
    """
    if not extraction_result:
        return "Searching for relevant items..."

    entities = extraction_result.get("entities", [])
    query_entities = extraction_result.get("query_entities", [])
    intents = extraction_result.get("intents", [])
    question_type = extraction_result.get("question_type", "list")

    parts = []

    # --- Focus on who/what is involved
    names = [qe["name"] for qe in query_entities if qe.get("name")]
    if names:
        parts.append(f"starring {', '.join(names)}")

    # --- Entity-based context
    if "genre" in entities:
        parts.append("by genre")
    if "company" in entities:
        parts.append("produced by specific studios")
    if "network" in entities:
        parts.append("available on specific networks")
    if "collection" in entities:
        parts.append("as part of a collection")

    # --- Date or rating
    if "date" in entities:
        parts.append("released after a certain year")
    if "rating" in entities:
        parts.append("with a minimum rating")

    # --- Base media type
    if "tv" in entities and "movie" not in entities:
        media_type = "TV shows"
    elif "movie" in entities and "tv" not in entities:
        media_type = "movies"
    else:
        media_type = "movies or TV shows"

    # --- Intent flavor
    if any(i.startswith("trending") for i in intents):
        flavor = "Trending"
    elif any(i.startswith("recommendation") for i in intents):
        flavor = "Recommended"
    else:
        flavor = "Searching for"

    # --- Compose final explanation
    filters = ", ".join(parts)
    if filters:
        return f"{flavor} {media_type} {filters}."
    else:
        return f"{flavor} {media_type}."
    
def generate_relaxation_explanation(dropped_constraints: List[str]) -> str:
    """
    Generate a human-readable explanation of which constraints were relaxed.
    """
    if not dropped_constraints:
        return ""

    pieces = []
    mapping = {
        "with_people": "specific actors",
        "with_networks": "specific networks",
        "with_companies": "specific studios",
        "director_id": "specific directors",
        "with_genres": "specific genres",
        "primary_release_year": "specific release year",
        "vote_average.gte": "minimum rating",
        "with_runtime.gte": "minimum runtime",
        "with_runtime.lte": "maximum runtime"
    }

    for param in dropped_constraints:
        label = mapping.get(param, param)
        pieces.append(f"relaxed {label}")

    explanation = ", ".join(pieces)
    return f"⚠️ Note: Some filters were relaxed to find results ({explanation})."

# --- Phase 19 Addition: QueryExplanationBuilder ---

class QueryExplanationBuilder:
    @staticmethod
    def build_final_explanation(extraction_result, relaxed_parameters: list = None, fallback_used: bool = False) -> str:
        """
        Build a natural-language explanation based on planning and execution events.
        """
        query_entities = extraction_result.get("query_entities", []) or []
        explanation_parts = []

        def describe_entity(ent):
            """Return a natural-language phrase for an entity."""
            name = ent.get("name", "")
            ent_type = ent.get("type", "")
            role = ent.get("role", "actor").capitalize()

            if ent_type == "genre":
                return f"{name} genre"
            elif ent_type == "company":
                return f"produced by {name}"
            elif ent_type == "network":
                return f"aired on {name}"
            elif ent_type == "person":
                return f"{role} {name}"
            return name  # fallback for unknown types

        # 1. What was applied (entities involved)
        if query_entities:
            entity_descriptions = [describe_entity(ent) for ent in query_entities if describe_entity(ent)]
            if entity_descriptions:
                applied_summary = " and ".join(entity_descriptions)
                explanation_parts.append(f"Planned for {applied_summary}.")

        # 2. What was relaxed
        if relaxed_parameters:
            relaxed_parameters = sorted(set(relaxed_parameters))  # de-duplicate and sort
            if relaxed_parameters:
                if len(relaxed_parameters) == 1:
                    relaxed_text = relaxed_parameters[0]
                else:
                    relaxed_text = ", ".join(relaxed_parameters[:-1]) + f" and {relaxed_parameters[-1]}"
                explanation_parts.append(f"Relaxed constraints on {relaxed_text} to find matches.")

        # 3. Fallback notice
        if fallback_used:
            explanation_parts.append("Fallback discovery was used to broaden the search results.")

        # 4. Combine all parts
        if not explanation_parts:
            return "Performed a general search based on available information."

        return " ".join(explanation_parts)
