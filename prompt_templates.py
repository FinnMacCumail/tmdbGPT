PLAN_PROMPT = """
Generate API execution plans USING THESE AVAILABLE ENDPOINTS:
{api_context}

Follow these rules:
1. Only use endpoints listed above
2. Required parameters must match endpoint specs
3. Steps must be chained using $entity references

Current Query: {query}
Resolved Entities: {entities}
Intent: {intents}
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