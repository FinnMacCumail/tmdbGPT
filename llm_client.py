# llm_client.py

from openai import OpenAI
import os
import json

class OpenAILLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract_entities_and_intents(self, query: str) -> dict:
        
        prompt = f"""
        You are a TMDB assistant. Analyze the user's query and extract three components using this exact JSON schema:

        {{
        "intents": [
            "List of intents related to the user's request. Examples include:",
            "'search.multi', 'discovery.filtered', 'recommendation.similarity', 'trending.popular', 'details.movie', etc."
        ],
        "entities": [
            "List of general entity types mentioned or implied in the query.",
            "Use only: 'person', 'movie', 'tv', 'genre', 'keyword', 'company', 'collection', 'network', 'date', 'rating', 'language'"
        ],
        "query_entities": [
            {{
            "name": "Full name or title of a specific person, movie, keyword, etc.",
            "type": "Exact entity type: 'person', 'movie', 'tv', 'genre', 'keyword', 'company', 'collection', or 'network'"
            }}
        ]
        }}

        Guidelines:
        - Always include `query_entities` for specific names, actors, studios, genres, keywords, or titles mentioned.
        - Always assign a `type` to each `query_entity`. Use your best judgment based on the query.
        - Include all applicable `intents` and `entities`, even if no named `query_entity` is present.
        - Use lowercase values for all types and intents.
        - If a query is vague or exploratory, fall back to general types (e.g., 'movie', 'genre').
        - Do NOT include commentary — only respond with valid JSON.

        User Query:
        \"{query.strip()}\"

        Respond ONLY with valid JSON.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You extract structured TMDB intents and entities."},
                    {"role": "user", "content": prompt.strip()}
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            return {"intents": [], "entities": [], "query_entities": []}

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",  # or gpt-4 if you have access
            messages=[
                {"role": "system", "content": "You are a helpful API assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
