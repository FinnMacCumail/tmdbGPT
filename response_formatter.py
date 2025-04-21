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
def format_count_summary(state) -> list:
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
    return [{"type": "count_summary", "text": output}]

@register_renderer("ranked_list")
def format_ranked_list(state) -> list:
    return ResponseFormatter.format_responses(state.responses)


@register_renderer("summary")
def format_summary(state) -> list:
    profiles = [r for r in state.responses if r.get("type") == "person_profile"]
    if profiles:
        name = profiles[0].get("title", "Unknown")
        overview = profiles[0].get("overview", "")
        return [f"👤 {name}: {overview}"]
    return ResponseFormatter.format_responses(state.responses)
