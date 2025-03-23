import re
class QueryClassifier:
    QUERY_PATTERNS = {
        "trending": r"\b(trending|popular|top)\b",
        "filmography": r"\b(directed by|starring|featuring)\b",
        "filtered_search": r"\b(from|with|rated|released|between)\b",
        "comparison": r"\b(most|best|highest|top-grossing)\b",
        "similarity": r"\b(similar to|like|recommendations)\b"
    }

    def classify(self, query: str) -> str:
        for pattern_type, regex in self.QUERY_PATTERNS.items():
            if re.search(regex, query, re.I):
                return pattern_type
        return "general_info"