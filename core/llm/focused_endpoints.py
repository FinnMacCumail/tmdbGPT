# core/llm/focused_endpoints.py

import json
from core.llm.llm_client import OpenAILLMClient


def get_focused_endpoints(query: str, symbolic_matches: list, question_type: str = None) -> list:
    """
    Ask the LLM to select relevant endpoints from symbolic_matches,
    given the user's query and extracted question_type.
    """
    endpoint_descriptions = [
        {
            "path": m.get("path") or m.get("endpoint"),
            "media_type": m.get("media_type", "any"),
            "supported_intents": m.get("intents", []),
            "consumes_entities": m.get("consumes_entities", [])
        }
        for m in symbolic_matches
    ]

    prompt = f"""
You're a TMDB planner assistant. Given the user's query and the extracted question type '{question_type}', choose the relevant endpoints needed to accurately fulfill the user's request.

Query:
"{query}"

Question Type:
- "count": Numeric totals or counts.
- "summary": Brief descriptions or bios.
- "timeline": Chronological sequences of events or works.
- "comparison": Side-by-side comparisons.
- "fact": Direct factual answers.
- "list": Curated lists or filtered recommendations.

Candidate Endpoints:
{json.dumps(endpoint_descriptions, indent=2)}

Respond with JSON:
{{ "recommended_endpoints": ["/endpoint/path", ...] }}
    """

    try:
        llm = OpenAILLMClient()
        response = llm.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        return result.get("recommended_endpoints", [])
    except Exception as e:
        print(f"⚠️ LLM planner failed: {e}")
        return []
