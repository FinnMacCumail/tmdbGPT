import re

class QueryClassifier:
    QUERY_PATTERNS = {
        "trending": r"\b(trending|popular|top|hot|trend)\b",
        "filmography": r"\b(directed by|starring|featuring|acted in|filmography)\b",
        "filtered_search": r"\b(from|with|rated|released|between|genre|year|decade)\b",
        "comparison": r"\b(most|best|highest|top-grossing|worst|lowest)\b",
        "similarity": r"\b(similar to|like|recommendations|related to)\b",
        "awards": r"\b(awards|oscars|emmys|nominations|won)\b",
        "financial": r"\b(box office|revenue|budget|gross|earning)\b",
        "multistep": r"\b(then|after that|followed by|next)\b"
    }

    QUERY_PATTERN_PRIORITY = [
        "multistep", "financial", "awards", "similarity",
        "comparison", "filtered_search", "trending", "filmography"
    ]

    def classify(self, query: str) -> str:
        matches = []
        for pattern_type, regex in self.QUERY_PATTERNS.items():
            if re.search(regex, query, re.I):
                matches.append(pattern_type)
        
        for p in self.QUERY_PATTERN_PRIORITY:
            if p in matches:
                return p
        return "general_info"