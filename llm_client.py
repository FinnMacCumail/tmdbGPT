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
        ],
            "question_type": "count | summary | timeline | comparison | fact | list",
            "response_format": "count_summary | summary | timeline | side_by_side | ranked_list"
        }}

        Guidelines:
        - Always include `query_entities` for specific names, actors, studios, genres, keywords, or titles mentioned.
        - Always assign a `type` to each `query_entity`. Use your best judgment based on the query.
        - Include all applicable `intents` and `entities`, even if no named `query_entity` is present.
        - Use lowercase values for all types and intents.
        - If a query is vague or exploratory, fall back to general types (e.g., 'movie', 'genre').
        - Include rating values like 'above 7.5' as {{ "name": "7.5", "type": "rating" }}
        - Include year references like 'from 2023' as {{ "name": "2023", "type": "date" }}
        - Use 'count_summary' for queries like "how many movies..."
        - Use 'timeline' for "what was the first... what came after..."
        - Use 'side_by_side' for "which is better, X or Y?"
        - Use 'summary' for bios or overviews.
        - Default to 'ranked_list' for recommendations or results.
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
                temperature=0,
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
                    # Heuristic: if 'directed' appears, assign first match to director
                    if "directed by" in query.lower():
                        ent["role"] = "director" if i == 0 else "cast"
                    else:
                        ent["role"] = "cast" if i == 0 else "director"
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
        
    def get_focused_endpoints(self, query, symbolic_matches, question_type=None):
        endpoint_descriptions = []
        for match in symbolic_matches:
            endpoint_descriptions.append({
                "path": match.get("path") or match.get("endpoint"),
                "media_type": match.get("media_type", "any"),
                "supported_intents": match.get("intents", []),
                "consumes_entities": match.get("consumes_entities", [])
            })

        # Debugging information
        print(f"\n🔍 [LLM Endpoint Planner Debugging] 🔍")
        print(f"→ Query: '{query}'")
        print(f"→ Question Type: '{question_type}'")
        print(f"→ Candidate Endpoints (before filtering): {json.dumps(endpoint_descriptions, indent=2)}\n")

        prompt = f"""
    You are a TMDB symbolic planning assistant. Your task is to recommend a concise set of TMDB endpoints required to answer the user's question based on the extracted question type and intents.

    Question Type: "{question_type}"

    Question types explained:
    - "count": User expects numeric counts (e.g., number of movies an actor has been in).
    - "summary": User expects a brief overview or bio.
    - "timeline": User expects chronologically ordered results (e.g., career timeline).
    - "comparison": User expects side-by-side comparisons.
    - "fact": User expects a simple factual answer.
    - "list": User expects a curated list of items.

    Given the user's query and these candidate endpoints, respond ONLY with endpoints directly useful to answer the query according to the extracted question type.

    User Query:
    "{query}"

    Candidate Endpoints:
    {json.dumps(endpoint_descriptions, indent=2)}

    Respond with JSON:
    {{ "recommended_endpoints": ["/person/{{person_id}}/movie_credits", ...] }}
    """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # More detailed debugging
            print(f"\n✅ [LLM Planner Response Debugging] ✅")
            print(f"Raw LLM Output: {content}")
            recommended_endpoints = result.get('recommended_endpoints', [])
            print(f"Extracted recommended endpoints: {recommended_endpoints}\n")

            return recommended_endpoints

        except Exception as e:
            print(f"⚠️ [LLM Planner Exception] Failed to get focused endpoints: {e}")
            return []



    # The method below is redundant - it and the classes that use it need to be cleaned!
    # def generate_response(self, prompt: str) -> str:
    #     response = self.client.chat.completions.create(
    #         model="gpt-4-turbo",
    #         messages=[
    #             {"role": "system", "content": "You are a helpful API assistant."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.3,
    #     )
    #     return response.choices[0].message.content