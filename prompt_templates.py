PLAN_PROMPT = """
Generate API plan based on query type and these rules. Generate a valid JSON API execution plan for the query provided.:
1. Use $entity syntax for resolved IDs
2. Prioritize generic_search endpoints
3. Include both entity resolution and data retrieval steps

Combine search and details:
    1. Search for entity using /search/{{resource}}
    2. Get details using /{{resource}}/{{id}}

Current Entities: {entities}
Query: {query}

Example JSON:
{{
  "plan": [
    {{
      "step_id": "unique_id",
      "endpoint": "/endpoint/{{param}}",
      "method": "GET",
      "parameters": {{"param": "$entity"}}
    }}
  ]
}}
"""


# PROMPT_TEMPLATES = {
#     "trending": """Use trending endpoints:
#     - /trending/{{media_type}}/{{time_window}}
#     Required parameters: media_type=movie|tv, time_window=day|week""",
    
#     "filtered_search": """Use discover endpoint with filters:
#     - /discover/movie?with_genres=GENRE_ID&primary_release_year=YEAR
#     - Sort by: vote_average.desc, revenue.desc""",
    
#     "similarity": """Use recommendations system:
#     1. First find base entity ID
#     2. Then use /movie/{{id}}/recommendations""",
    
#     "financial": """Combine movie details with financial data:
#     1. Get movie ID
#     2. Access /movie/{{id}}/revenue""",
    
#     "awards": """Use awards endpoints:
#     - /movie/{{id}}/awards
#     - /person/{{id}}/awards"""
# }

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