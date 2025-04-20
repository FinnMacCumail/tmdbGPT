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