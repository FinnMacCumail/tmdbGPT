# core/planner/extractor.py

import json
from core.llm.llm_client import OpenAILLMClient
from core.llm.role_inference import (
    infer_role_for_entity,
    infer_role_from_query,
    infer_media_type_from_query
)


def extract_entities_and_intents(query: str) -> dict:
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
            "type": "Exact entity type: 'person', 'movie', 'tv', 'genre', 'keyword', 'company', 'collection', 'network', 'rating', or 'date'",
            "role": "Optional — Only for 'person' type. Role such as 'cast', 'director', 'writer', 'producer', or 'composer'. Default to 'cast' if uncertain."
            }}
        ],
            "question_type": "count | summary | timeline | comparison | fact | list",
            "response_format": "count_summary | summary | timeline | side_by_side | ranked_list"
        }}

        Guidelines:
        - Always include `query_entities` for specific names, actors, studios, genres, keywords, or titles mentioned.
        - Always assign a `type` to each `query_entity`. Use your best judgment based on the query.
        - Include all applicable `intents` and `entities`, even if no named `query_entity` is present.
        - For 'person' entities, if you can infer the role from context (e.g., starring, directed by, written by), include a 'role' field.
        - If the role is unclear, default role to 'cast'.
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
        llm = OpenAILLMClient()
        response = llm.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system",
                    "content": "You extract structured TMDB intents and entities."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        result = json.loads(content)  # ✅ Safe JSON parsing

        # ✅ Safeguard extracted fields
        # Ensure standard fields
        result.setdefault("query_entities", [])
        result.setdefault("intents", [])
        result.setdefault("entities", [])
        result.setdefault("question_type", "summary")
        result.setdefault("response_format", "ranked_list")

        # phase 19.9 - Media Type Enforcement Baseline
        result["media_type"] = infer_media_type_from_query(query)

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
        dynamic_services = {"netflix", "amazon prime", "prime video",
                            "hulu", "disney+", "apple tv", "peacock", "paramount+"}
        always_network_services = {"hbo", "starz"}

        # phase 20 - Role-Aware Multi-Entity Planning and Execution
        for ent in result.get("query_entities", []):
            if ent.get("type") == "person" and "role" not in ent:
                inferred_role = infer_role_for_entity(ent["name"], query)
                ent["role"] = inferred_role

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
                        ent["type"] = corrected_type
                        if corrected_type not in result["entities"]:
                            result["entities"].append(corrected_type)

        return result

    except Exception as e:
        return {
            "intents": [],
            "entities": [],
            "query_entities": [],
            "question_type": "summary",
            "response_format": "summary",
            "media_type": "both"
        }
