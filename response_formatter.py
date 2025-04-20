from typing import List, Dict

class ResponseFormatter:
    @staticmethod
    def format_responses(responses: List[dict]) -> List[str]:
        formatted = []

        def tag_for_score(score: float) -> str:
            if score >= 1.0:
                return "🎯"
            elif score >= 0.75:
                return "✅"
            elif score >= 0.5:
                return "➖"
            else:
                return "⚠️"

        for r in responses:
            if isinstance(r, str):
                formatted.append(r)
            elif isinstance(r, dict) and r.get("type") == "movie_summary":
                title = r.get("title", "Untitled")
                overview = r.get("overview", "")
                score = r.get("final_score", 1.0)
                source = r.get("source", "")
                tag = tag_for_score(score)
                line = f"{tag} {title}: {overview}"
                if source:
                    line += f"  (from {source})"
                formatted.append(line)
            else:
                formatted.append(f"📎 {str(r)}")

        return sorted(formatted, reverse=True)  # sorted alphabetically for now
