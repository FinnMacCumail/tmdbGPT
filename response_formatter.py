from typing import List, Dict

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
            if "_relaxed_" in source:
                return "♻️"
            return ""

        enriched = []
        for r in responses:
            if isinstance(r, str):
                enriched.append({"line": r, "score": 0.3})
            elif isinstance(r, dict) and r.get("type") == "movie_summary":
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
    
RESPONSE_RENDERERS = {}

def register_renderer(name):
    def decorator(func):
        RESPONSE_RENDERERS[name] = func
        return func
    return decorator

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
    output = f"🎬 {name} has appeared in {movie_count} movie(s) and {tv_count} TV show(s)."
    return {
        "response_format": "count_summary",
        "question_type": "count",
        "entity": name,
        "text": output
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
    profiles = [r for r in state.responses if r.get("type") == "person_profile"]
    if profiles:
        name = profiles[0].get("title", "Unknown")
        overview = profiles[0].get("overview", "")
        return {
            "response_format": "summary",
            "question_type": "summary",
            "entity": name,
            "text": f"👤 {name}: {overview}"
        }

    return {
        "response_format": "summary",
        "question_type": "summary",
        "text": "⚠️ No summary available."
    }

@register_renderer("timeline")
def format_timeline(state) -> list:
    timeline_entries = []

    for item in state.responses:
        if item.get("type") != "movie_summary":
            continue

        title = item.get("title", "Untitled")
        overview = item.get("overview", "No synopsis available.")
        source = item.get("source", "")
        score = item.get("final_score", 1.0)

        # Try to extract year from known TMDB keys or fallback to regex
        year = None
        if "release_date" in item:
            year = item["release_date"][:4]
        else:
            import re
            match = re.search(r"(19|20)\d{2}", overview)
            if match:
                year = match.group(0)

        timeline_entries.append({
            "title": title,
            "overview": overview,
            "release_year": int(year) if year and year.isdigit() else None,
            "source": source,
            "score": score
        })

    # Sort by year if present, otherwise by score
    timeline_entries.sort(key=lambda x: x.get("release_year") or 3000)

    return {
        "response_format": "timeline",
        "question_type": "timeline",
        "entity": state.extraction_result.get("query_entities", [{}])[0].get("name", ""),
        "entries": timeline_entries
    }

@register_renderer("comparison")
def format_comparison(state) -> dict:
    query_entities = state.extraction_result.get("query_entities", [])
    if len(query_entities) != 2:
        return {"type": "comparison", "error": "Comparison requires exactly two entities."}

    left_id = query_entities[0].get("resolved_id")
    right_id = query_entities[1].get("resolved_id")
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

        if str(left_id) in src:
            left_entries.append(entry)
        elif str(right_id) in src:
            right_entries.append(entry)

    # Sort entries by score and limit to top 3
    left_entries.sort(key=lambda x: x["score"], reverse=True)
    right_entries.sort(key=lambda x: x["score"], reverse=True)

    return {
        "response_format": "comparison",
        "question_type": "comparison",
        "left": { "name": left_name, "entries": left_entries[:3] },
        "right": { "name": right_name, "entries": right_entries[:3] }
    }

