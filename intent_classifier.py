# intent_classifier.py
from openai import OpenAI
from typing import List, Dict
import re
import json

class IntentClassifier:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.intent_labels = [
            "biographical", "discovery", "comparative", 
            "temporal", "multimedia", "credits", 
            "recommendation", "popularity", "combined"
        ]
        
    def classify(self, query: str) -> Dict:
        """Classify query using LLM with fallback to regex"""
        llm_result = self._llm_classification(query)
        if not llm_result:
            return self._rule_based_classification(query)
        return llm_result

    def _llm_classification(self, query: str) -> Dict:
        prompt = f"""Analyze this movie/TV query and respond with JSON:
        {{
            "primary_intent": [{self.intent_labels}],
            "secondary_intents": [{self.intent_labels}],
            "required_operations": ["search", "discover", "compare", "filter_by_year", ...],
            "implied_entities": ["person", "movie", "genre", "year", ...]
        }}

        Query: {query}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except:
            return {}

    def _rule_based_classification(self, query: str) -> Dict:
        patterns = {
            "biographical": r"\b(starring|directed by|director|actor)\b",
            "comparative": r"\b(vs|versus|compare|difference between)\b",
            "temporal": r"\b(recent|oldest|newest|from \d{4}s?)\b",
            "multimedia": r"\b(posters|images|trailers|photos)\b"
        }
        
        return {
            "primary_intent": next(
                (intent for intent, pattern in patterns.items() 
                 if re.search(pattern, query, re.I)), "unknown"),
            "secondary_intents": [
                intent for intent, pattern in patterns.items() 
                if re.search(pattern, query, re.I)
            ]
        }