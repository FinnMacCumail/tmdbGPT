PLAN_PROMPT = """Generate API plan based on query type and these rules. Generate a valid JSON API execution plan for the query provided.:
1. Use $entity syntax for resolved IDs
2. Prioritize {query_type} endpoints
3. Include both entity resolution and data retrieval steps

{template_hint}  # 🚨 Missing in context
Current Entities: {entities}
Query: {query}"""

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
    "filmography": """Use person movie credits endpoint:
    1. First find person ID using /search/person
    2. Then use /person/{{person_id}}/movie_credits""",
    
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