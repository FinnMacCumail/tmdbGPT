import re
from typing import Dict

class QueryClassifier:
    INTENT_PATTERNS = {
        "filmography": [
            r"\b(directed by|movies by|films of|filmography of|starring|acted by|movies featuring)\b"
        ],
        "trending": [
            r"\b(trending|popular|top|most watched|currently popular|what's hot|top rated|popular movies|popular films)\b"
        ],
        "search": [
            r"\b(find|search for|look up|movies like|similar to|show me)\b"
        ]
    }

    def classify(self, query: str) -> Dict:
        """Dynamically classify query intent using regex patterns"""
        query = query.lower()
        matched_intents = []

        for intent, patterns in self.INTENT_PATTERNS.items():
            if any(re.search(pattern, query) for pattern in patterns):
                matched_intents.append(intent)

        return {
            "primary_intent": matched_intents[0] if matched_intents else "generic_search",
            "secondary_intents": matched_intents[1:]
        }