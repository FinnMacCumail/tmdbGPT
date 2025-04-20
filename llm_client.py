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
            "type": "Exact entity type: 'person', 'movie', 'tv', 'genre', 'keyword', 'company', 'collection', 'network', 'rating', or 'date'"
            }}
        ]
        }}

        Guidelines:
        - Always include `query_entities` for specific names, actors, studios, genres, keywords, or titles mentioned.
        - Always assign a `type` to each `query_entity`. Use your best judgment based on the query.
        - Include all applicable `intents` and `entities`, even if no named `query_entity` is present.
        - Use lowercase values for all types and intents.
        - If a query is vague or exploratory, fall back to general types (e.g., 'movie', 'genre').
        - Include rating values like 'above 7.5' as {{ "name": "7.5", "type": "rating" }}
        - Include year references like 'from 2023' as {{ "name": "2023", "type": "date" }}
        - Do NOT include commentary — only respond with valid JSON.

        User Query:
        "{query.strip()}"

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
            result = json.loads(content)  # ✅ Safe JSON parsing

            # ✅ Fallback correction for known streaming services
            streaming_services = {
                "netflix": "company",
                "amazon prime": "company",
                "prime video": "company",
                "hulu": "company",
                "disney+": "company",
                "apple tv": "company",
                "peacock": "company",
                "paramount+": "company",
                "hbo": "network",
                "starz": "network"
            }

            for ent in result.get("query_entities", []):
                name_lower = ent.get("name", "").strip().lower()
                for keyword, corrected_type in streaming_services.items():
                    if keyword in name_lower and ent.get("type") != corrected_type:
                        print(f"🔁 Correcting '{ent['name']}' type: {ent['type']} → {corrected_type}")
                        ent["type"] = corrected_type
                        if corrected_type not in result.get("entities", []):
                            result["entities"].append(corrected_type)

            # ✅ Inject role tags (cast/director) based on wording or fallback heuristic
            person_entities = [e for e in result.get("query_entities", []) if e.get("type") == "person"]            
            # Assign roles using better fallback logic
            for i, ent in enumerate(person_entities):
                name = ent["name"].lower()
                if "directed by" in query.lower() and name in query.lower():
                    ent["role"] = "director"
                elif any(kw in query.lower() for kw in ["starring", "featuring", "actor", "acted by"]) and name in query.lower():
                    ent["role"] = "cast"
                elif len(person_entities) == 2:
                    # Fallback: assume 1st = director, 2nd = cast
                    ent["role"] = "director" if i == 0 else "cast"
                else:
                    # Safe fallback
                    ent["role"] = "cast"

            return result

        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            return {
                "intents": [],
                "entities": [],
                "query_entities": []
            }

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful API assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content