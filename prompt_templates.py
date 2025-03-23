PLAN_PROMPT = """Generate API plan based on query type and these rules:
1. Use $entity syntax for resolved IDs
2. Prioritize {query_type} endpoints
3. Include both entity resolution and data retrieval steps

{template_hint}

Current Entities: {entities}
Query: {query}"""

PROMPT_TEMPLATES = {
    "trending": """Use trending endpoints:
    - /trending/{{media_type}}/{{time_window}}
    Required parameters: media_type=movie|tv, time_window=day|week""",
    
    "filtered_search": """Use discover endpoint with filters:
    - /discover/movie?with_genres=GENRE_ID&primary_release_year=YEAR
    - Sort by: vote_average.desc, revenue.desc""",
    
    "similarity": """Use recommendations system:
    1. First find base entity ID
    2. Then use /movie/{{id}}/recommendations""",
    
    "financial": """Combine movie details with financial data:
    1. Get movie ID
    2. Access /movie/{{id}}/revenue""",
    
    "awards": """Use awards endpoints:
    - /movie/{{id}}/awards
    - /person/{{id}}/awards"""
}

FALLBACK_PROMPT = """Generate response using:
- Resolved IDs: {entities}
- Format: {format_hint}"""