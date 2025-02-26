import openai
import requests
import json
import os
from chromadb import PersistentClient
from dotenv import load_dotenv
from endMap import TMDB_API_MAP, handle_tmdb_dispatcher # Import API mapping and dispatcher

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ChromaDB Configuration (Persistent Storage)
CHROMA_DB_PATH = "/home/ola/ollamadev/funcall/embeddings/chroma_db"
chroma_client = PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

# OpenAI System Message
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

# OpenAI Function Definition for Dynamic TMDB Queries
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
                        "enum": list(TMDB_API_MAP.keys()),
                        "description": "The type of request (Movies, People, TV, etc.)."
                    },
                    "action": {
                        "type": "string",
                        "enum": list(set(action for cat in TMDB_API_MAP.values() for action in cat)),
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

# Function to Retrieve Best API Call from ChromaDB
def retrieve_best_tmdb_api_call(user_query):
    """Retrieve the best TMDB API call based on user input."""
    search_results = collection.query(
        query_texts=[user_query], 
        n_results=1
    )
    
    # Debugging: Print the retrieved metadata structure
    print(f"🛠️ Debug: ChromaDB Metadata Retrieved -> {search_results['metadatas']}")

    if not search_results["metadatas"] or not search_results["metadatas"][0]:
        return None, 0  # No valid match

    best_match_metadata = search_results["metadatas"][0]
    
    # If best_match_metadata is a list, get the first dictionary
    if isinstance(best_match_metadata, list):
        best_match_metadata = best_match_metadata[0]  # Extract first dictionary

    best_api_call = json.loads(best_match_metadata.get("solution", "{}"))  # Now safely access .get()
    match_score = search_results["distances"][0] if "distances" in search_results else 0

    return best_api_call, match_score

# User Query Processing
def handle_user_query(user_input):
    """Determines the correct API call and retrieves data dynamically."""

    # Try ChromaDB first
    best_api_call, match_score = retrieve_best_tmdb_api_call(user_input)

    # If a strong match is found in ChromaDB, use it
    if best_api_call and match_score > 0.7:
        print(f"🛠️ Debug: Using Cached API Call - Match Score: {match_score}")
        return handle_tmdb_dispatcher(best_api_call)

    # Otherwise, use OpenAI to generate a new API call dynamically
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


# API Response Formatter
def format_tmdb_response(response):
    """Dynamically formats any TMDB API response based on available fields."""
    
    if not response or "results" not in response or not response["results"]:
        return "❌ No results found."

    output = "📌 **Results:**\n"

    # Automatically detect relevant fields for each result item
    for item in response["results"][:10]:  # Show top 10 results dynamically
        item_data = []

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
