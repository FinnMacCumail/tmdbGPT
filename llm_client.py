# llm_client.py

from openai import OpenAI
import os
import json

# candidate for redundancy
# 🔹 NEW: Simple inlined Role Lexicon
ROLE_ALIASES = {
    "directed by": "director",
    "produced by": "producer",
    "written by": "writer",
    "screenplay by": "writer",
    "script by": "writer",
    "music by": "composer",
    "scored by": "composer",
    "starring": "cast",
    "acted in": "cast",
    "featuring": "cast",
    "performance by": "cast",
}

# candidate for redundancy
def infer_role_from_query(query: str) -> str:
    query_lower = query.lower()
    for phrase, role in ROLE_ALIASES.items():
        if phrase in query_lower:
            return role
    return "cast"  # Safe fallback

# phase 20 - Role-Aware Multi-Entity Planning and Execution
def infer_role_for_entity(name: str, query: str) -> str:
    query_lower = query.lower()
    name_lower = name.lower()

    # Find where the name appears
    idx = query_lower.find(name_lower)
    if idx == -1:
        return "cast"  # fallback

    # Look around the name ±20 characters
    window = 20
    start = max(0, idx - window)
    end = min(len(query_lower), idx + len(name_lower) + window)
    surrounding_text = query_lower[start:end]

    for phrase, role in ROLE_ALIASES.items():
        if phrase in surrounding_text:
            return role

    return "cast"  # fallback if no phrase found


def infer_media_type_from_query(query: str) -> str:
    query_lower = query.lower()
    if "tv show" in query_lower or "series" in query_lower:
        return "tv"
    elif "movie" in query_lower or "film" in query_lower:
        return "movie"
    else:
        return "both"

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

            # ✅ Safeguard extracted fields
            result.setdefault("query_entities", [])
            result.setdefault("intents", [])
            result.setdefault("entities", [])
            result.setdefault("question_type", "summary")
            result.setdefault("response_format", "summary")

            # phase 19.9 - Media Type Enforcement Baseline
            result["media_type"] = infer_media_type_from_query(query)
            print(f"🎥 Inferred media type: {result['media_type']}")

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

            # Dynamic streaming service correction
            dynamic_services = {"netflix", "amazon prime", "prime video", "hulu", "disney+", "apple tv", "peacock", "paramount+"}
            always_network_services = {"hbo", "starz"}

            # phase 20 - Role-Aware Multi-Entity Planning and Execution
            for ent in result.get("query_entities", []):
                if entity.get("type") == "person" and "role" not in entity:
                    inferred_role = infer_role_for_entity(entity["name"], query)
                    entity["role"] = inferred_role
                    print(f"🔎 Smarter role inferred for '{entity['name']}': {inferred_role}")
                name_lower = ent.get("name", "").strip().lower()

                for keyword in streaming_services:
                    if keyword in name_lower:
                        if keyword in dynamic_services:
                            # ✅ Correct logic:
                            if result.get("media_type") == "tv":
                                corrected_type = "network"
                            else:
                                corrected_type = "company"
                        elif keyword in always_network_services:
                            corrected_type = "network"
                        else:
                            corrected_type = streaming_services[keyword]

                        if ent.get("type") != corrected_type:
                            print(f"🔁 Correcting '{ent['name']}' type dynamically: {ent.get('type')} → {corrected_type}")
                            ent["type"] = corrected_type
                            if corrected_type not in result["entities"]:
                                result["entities"].append(corrected_type)

            # ✅ Fallback: Ensure every person has a role, even if LLM missed it
            for entity in result.get("query_entities", []):
                if entity.get("type") == "person" and "role" not in entity:
                    inferred_role = infer_role_from_query(query)
                    entity["role"] = inferred_role
                    print(f"🔎 Inferred missing role for '{entity['name']}': {inferred_role}")

            
            return result

        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            return {
                "intents": [],
                "entities": [],
                "query_entities": []
            }
        
    def get_focused_endpoints(self, query, symbolic_matches, question_type=None):
        endpoint_descriptions = [
            {
                "path": m.get("path") or m.get("endpoint"),
                "media_type": m.get("media_type", "any"),
                "supported_intents": m.get("intents", []),
                "consumes_entities": m.get("consumes_entities", [])
            }
            for m in symbolic_matches
        ]

        # Verbose debugging
        print(f"\n🔎 [Debug] Query: '{query}'")
        print(f"🔎 [Debug] Question Type: '{question_type}'")
        print(f"🔎 [Debug] Candidate Endpoints: {json.dumps(endpoint_descriptions, indent=2)}")

        prompt = f"""
    You're a TMDB planner assistant. Given the user's query and the extracted question type '{question_type}', choose the relevant endpoints needed to accurately fulfill the user's request.

    Question Type Context:
    - "count": Numeric totals or counts.
    - "summary": Brief descriptions or bios.
    - "timeline": Chronological sequences of events or works.
    - "comparison": Side-by-side comparisons.
    - "fact": Direct factual answers.
    - "list": Curated lists or filtered recommendations.

    Query:
    "{query}"

    Candidate Endpoints:
    {json.dumps(endpoint_descriptions, indent=2)}

    Respond with JSON:
    {{ "recommended_endpoints": ["/endpoint/path", ...] }}
    """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # More verbose debugging output
            recommended_endpoints = result.get('recommended_endpoints', [])
            print(f"\n✅ [Debug] Raw LLM Output: {content}")
            print(f"✅ [Debug] Recommended Endpoints: {recommended_endpoints}")

            return recommended_endpoints

        except Exception as e:
            print(f"⚠️ [Debug] LLM Planner Error: {e}")
            return []