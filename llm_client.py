# llm_client.py

from openai import OpenAI
import os
import json

class OpenAILLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract_entities_and_intents(self, query: str) -> dict:
        
        prompt = f"""
        Extract intents and entities from the user's query using this schema:

        {{
            "intents": ["recommendation.similarity", "recommendation.suggested", "discovery.filtered",
                        "discovery.genre_based", "discovery.temporal", "discovery.advanced",
                        "search.basic", "search.multi", "media_assets.image", "media_assets.video",
                        "details.movie", "details.tv", "credits.movie", "credits.tv", "credits.person",
                        "trending.popular", "trending.top_rated", "reviews.movie", "reviews.tv",
                        "collections.movie", "companies.studio", "companies.network"],
            "entities": ["movie", "tv", "person", "company", "network", "collection", "genre", "year", "keyword", "credit", "rating", "date"],
            "query_entities": ["names, titles or specific things directly mentioned by the user"]
        }}

        User Query: \"{query}\"
        Respond with ONLY valid JSON. No commentary:
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Extract intent and entity data as JSON."},
                    {"role": "user", "content": prompt.strip()}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            print(f"❌ LLM Extraction Failed: {e}")
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
