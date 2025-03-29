PLAN_PROMPT = """
Generate API execution plans using TMDB endpoints. Follow these rules:

1. Required structure for each step:
{{
  "step_id": "unique_name",
  "endpoint": "/api/path/{{parameter}}",
  "method": "GET",
  "parameters": {{
    "param": "$resolved_entity"
  }}
}}

2. Dynamic patterns (replace CAPS):
- Search: /search/ENTITY_TYPE?query=...
- Details: /ENTITY_TYPE/{{ENTITY_TYPE_ID}}
- Relationships: /ENTITY_TYPE/{{ENTITY_TYPE_ID}}/RELATIONSHIP

3. Examples:
=== Movie Example ===
{{
  "plan": [
    {{
      "step_id": "search_movie",
      "endpoint": "/search/movie",
      "method": "GET",
      "parameters": {{"query": "{query}"}}
    }},
    {{
      "step_id": "get_details",
      "endpoint": "/movie/{{movie_id}}",
      "method": "GET",
      "parameters": {{"movie_id": "$movie_id"}}
    }}
  ]
}}

=== TV Example ===
{{
  "plan": [
    {{
      "step_id": "discover_tv",
      "endpoint": "/discover/tv",
      "method": "GET",
      "parameters": {{
        "with_genres": "$genre_id",
        "first_air_date_year": "2023"
      }}
    }}
  ]
}}

Current Context:
- Query: {query}
- Resolved Entities: {entities}
- Intent Context: {intents}
- Available Relationships: credits, similar, recommendations
"""

PROMPT_TEMPLATES = {
    "filmography": """Use movie credits endpoint:
    1. Use resolved person_id
    2. Call /person/{person_id}/movie_credits
    Required parameters: person_id=$person_id""",
    
    "trending": """Use trending endpoints:
    - /trending/{{media_type}}/{{time_window}}
    Required parameters: media_type=movie|tv, time_window=day|week""",
    
    "search": """Use discover endpoint with filters:
    - /discover/movie?with_genres=GENRE_ID&primary_release_year=YEAR
    - Sort by: vote_average.desc, revenue.desc""",
    
    "generic_search": """Combine search and details:
    1. Search for entity using /search/{{resource}}
    2. Get details using /{{resource}}/{{id}}"""
}

DEFAULT_TEMPLATE = """Standard data retrieval:
1. Get basic entity details
2. Retrieve supplemental information"""

FALLBACK_PROMPT = """Generate response using:
- Resolved IDs: {entities}
- Format: {format_hint}"""