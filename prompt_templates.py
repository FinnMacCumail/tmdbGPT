# prompt_templates.py
PLAN_PROMPT = """Generate API plan with these rules:
1. Use $entity syntax for resolved IDs
2. Always include detail endpoints for existing entities
3. Never repeat search steps for resolved entities

Example when person_id exists:
{{
  "plan": [
    {{
      "step_id": 1,
      "description": "Get person details",
      "endpoint": "/person/{{person_id}}",
      "method": "GET",
      "parameters": {{"person_id": "$person_id"}},
      "operation_type": "data_retrieval"
    }}
  ]
}}

Current Entities: {entities}
Query: {query}"""

FALLBACK_PROMPT = """Generate response using:
- Resolved IDs: {entities}
- Format: Name, Biography, Known For"""