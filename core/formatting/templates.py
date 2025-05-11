from core.formatting.registry import register_renderer
from typing import List
import re


def generate_explanation(extraction_result) -> str:
    query_entities = extraction_result.get("query_entities", [])
    entity_descriptions = [e.get("name")
                           for e in query_entities if e.get("name")]
    if entity_descriptions:
        return f"No results found for: {', '.join(entity_descriptions)}"
    return "No relevant information found."


@register_renderer("fallback")
def format_fallback(state) -> dict:
    explanation = generate_explanation(state.extraction_result)
    return {
        "response_format": "summary",
        "question_type": state.question_type or "summary",
        "text": f"⚠️ {explanation}",
        "entries": [f"⚠️ {explanation}"]
    }


@register_renderer("count_summary")
def format_count_summary(state) -> dict:
    movie_count = 0
    tv_count = 0

    entity = state.extraction_result.get("query_entities", [{}])[0]
    name = entity.get("name", "This person")
    role = entity.get("role", "director").lower()
    role_label = {
        "cast": "actor", "actor": "actor",
        "director": "director",
        "writer": "screenwriter",
        "producer": "producer",
        "composer": "composer"
    }.get(role, role)

    for r in state.responses:
        if not isinstance(r, dict):
            continue
        if r.get("type") != "movie_summary":
            continue
        src = r.get("source", "")
        job = r.get("job", "").lower()
        if "movie_credits" in src and job == "director":
            movie_count += 1
        elif "tv_credits" in src and job == "director":
            tv_count += 1

    text = f"🎬 {name} worked as a {role_label} in {movie_count} movie(s) and {tv_count} TV show(s)."
    return {
        "response_format": "count_summary",
        "question_type": "count",
        "entity": name,
        "text": text,
        "entries": [text]
    }


@register_renderer("ranked_list")
def format_ranked_list(state, include_debug=False):
    formatted = []
    for idx, item in enumerate(state.responses):
        if not isinstance(item, dict):
            continue
        line = f"{idx+1}. {item.get('title') or item.get('name')}"
        if include_debug and "_provenance" in item:
            prov = item["_provenance"]
            matched = ', '.join(prov.get("matched_constraints", []))
            relaxed = ', '.join(prov.get("relaxed_constraints", []))
            validated = ', '.join(prov.get("post_validations", []))
            line += f"  [matched: {matched} | relaxed: {relaxed} | validated: {validated}]"
        formatted.append(line)
    return formatted


@register_renderer("summary")
def format_summary(state) -> dict:
    responses = state.responses
    entries = []
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
        emoji = SUMMARY_TYPES.get(r_type, "🔹")
        title = r.get("title", "Untitled")
        overview = r.get("overview", "No summary available.")
        entries.append(f"{emoji} {title}: {overview.strip()}")

    return {
        "response_format": "summary",
        "question_type": state.extraction_result.get("question_type", "summary"),
        "entries": entries or ["⚠️ No summary available."]
    }


@register_renderer("timeline")
def format_timeline(state) -> dict:
    entries = []
    for item in state.responses:
        if not isinstance(item, dict) or item.get("type") != "movie_summary":
            continue
        title = item.get("title", "Untitled")
        overview = item.get("overview", "No synopsis available.")
        source = item.get("source", "")
        score = item.get("final_score", 1.0)
        year = item.get("release_date", "")[
            :4] if "release_date" in item else None
        if not year:
            match = re.search(r"(19|20)\\d{2}", overview)
            year = match.group(0) if match else None
        entries.append({
            "title": title,
            "overview": overview,
            "release_year": int(year) if year and year.isdigit() else None,
            "source": source,
            "score": score
        })

    entries.sort(key=lambda x: x.get("release_year") or 3000)
    name = state.extraction_result.get("query_entities", [{}])[
        0].get("name", "")
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
        return {
            "response_format": "comparison",
            "question_type": "comparison",
            "entries": ["⚠️ Comparison requires exactly two entities."]
        }

    left_id = str(query_entities[0].get("resolved_id"))
    right_id = str(query_entities[1].get("resolved_id"))
    left_name = query_entities[0].get("name", "Entity A")
    right_name = query_entities[1].get("name", "Entity B")

    left_entries = []
    right_entries = []

    for r in state.responses:
        if not isinstance(r, dict) or r.get("type") != "movie_summary":
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


class QueryExplanationBuilder:
    @staticmethod
    def build_final_explanation(extraction_result, relaxed_parameters: list = None, fallback_used: bool = False) -> str:
        query_entities = extraction_result.get("query_entities", []) or []
        explanation_parts = []

        def describe_entity(ent):
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
            return name

        if query_entities:
            entity_descriptions = [describe_entity(
                ent) for ent in query_entities if describe_entity(ent)]
            if entity_descriptions:
                applied_summary = " and ".join(entity_descriptions)
                explanation_parts.append(f"Planned for {applied_summary}.")

        if relaxed_parameters:
            relaxed_parameters = sorted(set(relaxed_parameters))
            if relaxed_parameters:
                if len(relaxed_parameters) == 1:
                    relaxed_text = relaxed_parameters[0]
                else:
                    relaxed_text = ", ".join(
                        relaxed_parameters[:-1]) + f" and {relaxed_parameters[-1]}"
                explanation_parts.append(
                    f"Relaxed constraints on {relaxed_text} to find matches.")

        if fallback_used:
            explanation_parts.append(
                "Fallback discovery was used to broaden the search results.")

        if not explanation_parts:
            return "Performed a general search based on available information."

        return " ".join(explanation_parts)
