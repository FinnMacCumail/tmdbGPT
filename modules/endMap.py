import openai
import requests
import json
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# TMDB Base URL
TMDB_BASE_URL = "https://api.themoviedb.org/3/"

# OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Centralized TMDB API Endpoints
TMDB_API_MAP = {
    "movies": {
        "popular": "movie/popular",
        "trending": "trending/movie/week",
        "search": "discover/movie",  
        "details": "movie/{id}",
        "recommendations": "movie/{id}/recommendations",
        "credits": "movie/{id}/credits"
    },
    "people": {
        "search": "search/person",
        "filmography": "person/{id}/movie_credits"
    },
    "tv": {
        "popular": "tv/popular",
        "trending": "trending/tv/week",
        "search": "search/tv",
        "details": "tv/{id}",
        "recommendations": "tv/{id}/recommendations",
        "credits": "tv/{id}/credits"
    }
}


# TMDB API Request Function
def call_tmdb_api(endpoint, params=None):
    """Make a request to the TMDB API."""
    
    url = f"{TMDB_BASE_URL}{endpoint}?api_key={TMDB_API_KEY}"
    
    if params:
        for key, value in params.items():
            url += f"&{key}={value}"

    response = requests.get(url)

    # Debugging output
    print(f"🔍 Debug: API Call - {url}")
    print(f"🔍 Debug: TMDB API Response: {response.json()}")

    return response.json()

# Dynamic API Request Dispatcher
def handle_tmdb_dispatcher(parameters):
    """Dynamically routes TMDB API requests based on OpenAI’s function call."""

    category = parameters.get("category")
    action = parameters.get("action")
    filters = parameters.get("filters", {})
    query = parameters.get("query", None)

    # Validate category & action
    if category not in TMDB_API_MAP or action not in TMDB_API_MAP[category]:
        return {"error": "❌ Invalid API request. Category or action not found."}

    # Retrieve the correct API endpoint
    endpoint = TMDB_API_MAP[category][action]

    # Handle ID-based requests
    if "{id}" in endpoint:
        if "id" not in filters:
            return {"error": "❌ Missing 'id' for this request."}
        endpoint = endpoint.replace("{id}", str(filters["id"]))

    # Construct API request dynamically
    params = {"query": query} if query else {}
    params.update(filters)  # Add additional filters

    return call_tmdb_api(endpoint, params)

# OpenAI Function Calling Setup
tmdb_functions = [
    {
        "type": "function",
        "function": {
            "name": "tmdb_dispatcher",
            "description": "Handles TMDB API requests for movies, people, and TV shows dynamically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": list(TMDB_API_MAP.keys()),  # Dynamically fetch categories
                        "description": "The type of request (Movies, People, TV, etc.)."
                    },
                    "action": {
                        "type": "string",
                        "enum": list(set(action for cat in TMDB_API_MAP.values() for action in cat)),  # All actions
                        "description": "The action to perform (search, trending, recommendations, etc.)."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'Sofia Coppola', 'Sci-Fi Movies')."
                    },
                    "filters": {
                        "type": "object",
                        "description": "Filters such as genre, year, popularity, etc.",
                        "additionalProperties": True
                    }
                },
                "required": ["category", "action"]
            }
        }
    }
]

# Optimized OpenAI System Message
system_message = """
You are a TMDB assistant that determines the correct API category and request type.

- **Movies:** Popular, Trending, Search, Details, Recommendations, Credits
- **People:** Search, Filmography, Credits
- **TV Shows:** Popular, Trending, Search, Details, Recommendations, Credits

🚨 **IMPORTANT:**
- **If the user requests movies with filters (e.g., action movies from 2022), use filters instead of query.**
- **If the user requests a director’s films, call the 'filmography' action.**

✅ **Examples of Correct API Calls:**
- "Find action movies from 2022" → `{ "category": "movies", "action": "search", "filters": { "with_genres": "28", "primary_release_year": "2022", "sort_by": "popularity.desc" } }`
- "How many films has Sofia Coppola directed?" → `{ "category": "people", "action": "filmography", "query": "Sofia Coppola" }`
"""

# User Query Handling
def handle_user_query(user_input):
    """Determines the correct API call dynamically from ChromaDB or OpenAI."""

    best_api_call, match_score = retrieve_best_tmdb_api_call(user_input)

    if best_api_call and match_score > 0.7:
        print(f"🛠️ Debug: Using Cached API Call - Match Score: {match_score}")
        return handle_tmdb_dispatcher(best_api_call)

    print("🛠️ Debug: No match in ChromaDB. Using OpenAI to generate API call.")

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_input}],
        tools=tmdb_functions,
        tool_choice="auto"
    )

    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        parameters = json.loads(tool_call.function.arguments)

        return handle_tmdb_dispatcher(parameters)

    return {"error": "❌ No valid tool call was made."}

def format_tmdb_response(response):
    """Dynamically formats any TMDB API response based on available fields."""

    if not response or "results" not in response or not response["results"]:
        return "❌ No results found."

    output = "📌 **Results:**\n"

    # Automatically detect relevant fields for each result item
    for item in response["results"][:10]:  # Show top 10 results dynamically
        item_data = []

        # Include any meaningful key-value pairs dynamically
        for key, value in item.items():
            if value and key not in ["id", "adult", "video", "backdrop_path", "poster_path", "original_language"]:
                item_data.append(f"**{key.replace('_', ' ').title()}**: {value}")

        output += "\n🎬 " + "\n".join(item_data) + "\n"

    return output


# Main Interactive Chatbot Loop
def main():
    """Main interactive loop for TMDB chatbot."""
    print("🎬 TMDB Chatbot: Ask me about movies, actors, or trending films!")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 🎬")
            break

        response = handle_user_query(user_input)

        # Dynamically format the API response
        print(format_tmdb_response(response))

# Run chatbot
if __name__ == "__main__":
    main()
