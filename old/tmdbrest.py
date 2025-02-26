import openai
import requests
import json
import os
from dotenv import load_dotenv

# Load API Keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Ensure you store TMDB API Key in .env
TMDB_API_KEY = "1370b21e6746a8aa1aff4b6e5b9fd92e"
TMDB_BASE_URL = "https://api.themoviedb.org/3/"

# OpenAI API Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🔹 Step 1: Classify User Query (LLM Preprocessing)
def classify_query(user_input):
    """Use OpenAI to classify the user's intent before making an API call."""

    classification_prompt = f"""
    Classify the following user query into a structured JSON object.

    User Query: "{user_input}"

    Categories:
    - "movies": General movie-related queries (e.g., "What are the trending movies?")
    - "people": Queries about directors, actors, or filmographies (e.g., "How many films has Sofia Coppola directed?")
    - "search": Searching for specific movies or persons (e.g., "Search for the movie Inception")

    Additional Fields:
    - "action": Defines the action (e.g., "popular", "trending", "credits", "details", "search").
    - "query": Extract the relevant search term (e.g., movie title or person name).

    Example Output:
    {{
        "category": "people",
        "action": "credits",
        "query": "Sofia Coppola"
    }}

    Return only a valid JSON object.
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": classification_prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# 🔹 Step 2: TMDB API Request Handler
def call_tmdb_api(endpoint, params=None):
    """Make a request to the TMDB API and return the response."""
    
    url = f"{TMDB_BASE_URL}{endpoint}?api_key={TMDB_API_KEY}"
    
    if params:
        for key, value in params.items():
            url += f"&{key}={value}"

    response = requests.get(url)
    
    return response.json()

# 🔹 Step 3: Fetch Directed Movies
def fetch_directed_movies(person_name):
    """Retrieve movies directed by a specific person using TMDB API."""
    
    # Step 1: Search for the person
    search_result = call_tmdb_api("search/person", {"query": person_name})

    if not search_result["results"]:
        return {"error": f"❌ No results found for '{person_name}'."}

    # Step 2: Get first match
    person_id = search_result["results"][0]["id"]
    
    # Step 3: Fetch movie credits
    movie_credits = call_tmdb_api(f"person/{person_id}/movie_credits")

    # Step 4: Filter directed movies
    directed_movies = [movie for movie in movie_credits.get("crew", []) if movie.get("job") == "Director"]
    num_directed = len(directed_movies)

    if num_directed == 0:
        return {"message": f"🎬 {person_name} has not directed any films according to TMDB."}

    # Step 5: Format the response
    response = f"\n🎬 **{num_directed} films directed by {person_name}:**\n"
    for movie in directed_movies:
        title = movie.get("title", "Unknown Title")
        release_year = movie.get("release_date", "Unknown Date")[:4]
        response += f" - {title} ({release_year})\n"

    return {"director_count": num_directed, "movies": directed_movies, "response": response}

# 🔹 Step 4: Define OpenAI Function Calling (Tool)
tmdb_dispatcher_function = {
    "name": "tmdb_dispatcher",
    "description": "Handles different types of TMDB API requests, including movies, people, and searches.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["movies", "people", "search"],
                "description": "The category of the TMDB request."
            },
            "action": {
                "type": "string",
                "enum": ["popular", "trending", "details", "search", "credits"],
                "description": "The specific action to perform."
            },
            "query": {
                "type": "string",
                "description": "A search query if applicable (e.g., movie name, actor name)."
            }
        },
        "required": ["category", "action"]
    }
}

# 🔹 Step 5: OpenAI Tool Calling (Function Execution)
def get_openai_response(user_input):
    """Classify user query and use OpenAI function calling to execute it."""

    # Step 1: Classify Query
    structured_query = classify_query(user_input)
    print("🛠 Debug: Classified Query →", structured_query)

    # Step 2: OpenAI Tool Calling
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Use the following structured intent to generate the correct function call."},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(structured_query)}
        ],
        tools=[{"type": "function", "function": tmdb_dispatcher_function}],
        tool_choice="auto"
    )

    return response

# 🔹 Step 6: Process OpenAI Response
def process_openai_response(response):
    """Extract and execute the correct function call from OpenAI's response."""
    
    if response.choices and response.choices[0].message.tool_calls:
        tool_calls = response.choices[0].message.tool_calls
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            parameters = json.loads(tool_call.function.arguments)

            if function_name == "tmdb_dispatcher":
                return handle_tmdb_dispatcher(parameters)

    return {"error": "❌ Invalid request"}

def search_tmdb_person(person_name):
    """Search TMDB for a person by name and return relevant details."""
    
    # Step 1: Search for the person
    search_result = call_tmdb_api("search/person", {"query": person_name})

    if not search_result["results"]:
        return {"error": f"❌ No results found for '{person_name}'."}

    # Step 2: Get the first result (most relevant)
    person = search_result["results"][0]
    person_id = person["id"]
    name = person["name"]
    known_for = ", ".join([movie["title"] for movie in person.get("known_for", []) if "title" in movie])

    # Step 3: Get additional details
    person_details = call_tmdb_api(f"person/{person_id}")

    bio = person_details.get("biography", "No biography available.")
    birth_date = person_details.get("birthday", "Unknown")
    death_date = person_details.get("deathday", None)
    place_of_birth = person_details.get("place_of_birth", "Unknown")
    
    # Step 4: Format response
    response = f"🎭 **{name}**\n\n"
    response += f"📅 Born: {birth_date} in {place_of_birth}\n"
    
    if death_date:
        response += f"🕊️ Passed away: {death_date}\n"

    response += f"\n📜 **Biography:**\n{bio[:500]}..."  # Show first 500 characters
    response += f"\n🎬 **Known for:** {known_for}" if known_for else ""

    return {"name": name, "bio": bio, "birth_date": birth_date, "known_for": known_for, "response": response}


# 🔹 Step 7: Handle TMDB Dispatcher Calls
def handle_tmdb_dispatcher(parameters):
    """Execute the correct TMDB function based on OpenAI's structured query."""
    
    category = parameters.get("category")
    action = parameters.get("action")
    query = parameters.get("query")

    if category == "movies":
        if action == "popular":
            return call_tmdb_api("movie/popular")
        elif action == "trending":
            return call_tmdb_api("trending/movie/week")

    elif category == "people":
        if action == "credits":
            return fetch_directed_movies(query)
        elif action == "search":
            return search_tmdb_person(query)

    elif category == "search":
        return call_tmdb_api("search/movie", {"query": query})

    return {"error": "❌ Could not process request."}

# 🔹 Step 8: Main User Interaction
def main():
    """Main chatbot loop."""
    
    print("🎬 TMDB Chatbot: Ask me about movies, actors, or trending films!")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 🎬")
            break

        response = get_openai_response(user_input)
        tmdb_data = process_openai_response(response)

        print("\n", tmdb_data.get("response", tmdb_data))

# Run the chatbot
if __name__ == "__main__":
    main()
