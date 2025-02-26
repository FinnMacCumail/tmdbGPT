import openai
import requests
import json
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage

# Load API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = "1370b21e6746a8aa1aff4b6e5b9fd92e"
TMDB_BASE_URL = "https://api.themoviedb.org/3/"

# OpenAI API Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🔹 SESSION MEMORY FOR MULTI-TURN CONTEXT
session_memory = {}

def store_session(user_id, query, response):
    """Store user query and response for follow-up context."""
    session_memory[user_id] = {"query": query, "response": response}

def retrieve_session(user_id):
    """Retrieve last user response for contextual follow-up questions."""
    return session_memory.get(user_id, {})

# 🔹 LLM QUERY ANALYSIS (Step 1)
def analyze_query(user_input):
    """Use OpenAI to determine what API calls are needed for the query."""
    
    prompt = f"""
    Analyze the following user query and break it down into a sequence of necessary API calls.

    Query: "{user_input}"

    Return a JSON array where each object represents an API call. Example:
    [
        {{
            "category": "people",
            "action": "search",
            "query": "Sofia Coppola"
        }},
        {{
            "category": "people",
            "action": "credits",
            "query": "Sofia Coppola"
        }}
    ]
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# 🔹 CALL TMDB API
def call_tmdb_api(endpoint, params=None):
    """Make a request to the TMDB API and return the response."""
    
    url = f"{TMDB_BASE_URL}{endpoint}?api_key={TMDB_API_KEY}"
    
    if params:
        for key, value in params.items():
            url += f"&{key}={value}"

    response = requests.get(url).json()
    
    if "results" not in response:
        return {"error": "❌ TMDB API returned an unexpected response."}

    return response

# 🔹 SEARCH FOR A PERSON IN TMDB
def search_tmdb_person(person_name):
    """Search TMDB for a person by name and return relevant details."""

    search_result = call_tmdb_api("search/person", {"query": person_name})

    if not search_result.get("results"):
        return {"error": f"❌ No results found for '{person_name}'."}

    person = search_result["results"][0]
    person_id = person["id"]
    name = person["name"]
    known_for = ", ".join([movie["title"] for movie in person.get("known_for", []) if "title" in movie])

    person_details = call_tmdb_api(f"person/{person_id}")

    if "biography" not in person_details:
        return {"error": f"❌ Could not retrieve details for '{name}'."}

    bio = person_details.get("biography", "No biography available.")
    birth_date = person_details.get("birthday", "Unknown")
    death_date = person_details.get("deathday", None)
    place_of_birth = person_details.get("place_of_birth", "Unknown")

    response = f"🎭 **{name}**\n📅 Born: {birth_date} in {place_of_birth}\n"
    
    if death_date:
        response += f"🕊️ Passed away: {death_date}\n"

    response += f"\n📜 **Biography:**\n{bio[:500]}..."
    response += f"\n🎬 **Known for:** {known_for}" if known_for else ""

    return {"name": name, "bio": bio, "birth_date": birth_date, "known_for": known_for, "response": response}

# 🔹 FETCH MOVIES DIRECTED BY A PERSON
def fetch_directed_movies(person_name):
    """Retrieve movies directed by a specific person using TMDB API."""
    
    search_result = call_tmdb_api("search/person", {"query": person_name})

    if not search_result["results"]:
        return {"error": f"❌ No results found for '{person_name}'."}

    person_id = search_result["results"][0]["id"]
    movie_credits = call_tmdb_api(f"person/{person_id}/movie_credits")

    directed_movies = [movie for movie in movie_credits.get("crew", []) if movie.get("job") == "Director"]
    num_directed = len(directed_movies)

    if num_directed == 0:
        return {"message": f"🎬 {person_name} has not directed any films."}

    response = f"\n🎬 **{num_directed} films directed by {person_name}:**\n"
    for movie in directed_movies:
        title = movie.get("title", "Unknown Title")
        release_year = movie.get("release_date", "Unknown Date")[:4]
        response += f" - {title} ({release_year})\n"

    return {"director_count": num_directed, "movies": directed_movies, "response": response}

# 🔹 EXECUTE API CALLS (Step 2)
def execute_api_calls(api_requests):
    """Dynamically execute multiple API requests based on query breakdown."""

    results = []
    for request in api_requests:
        category = request.get("category")
        action = request.get("action")
        query = request.get("query")

        if category == "movies":
            if action == "popular":
                results.append(call_tmdb_api("movie/popular"))
            elif action == "trending":
                results.append(call_tmdb_api("trending/movie/week"))

        elif category == "people":
            if action == "search":
                results.append(search_tmdb_person(query))
            elif action == "credits":
                results.append(fetch_directed_movies(query))

        elif category == "search":
            results.append(call_tmdb_api("search/movie", {"query": query}))

    return results

# 🔹 VALIDATE & REFINE RESPONSE (Step 3)
def validate_and_refine_response(api_results):
    """Use LLM to validate data completeness and refine the response."""

    validation_prompt = f"""
    Validate and refine the following API responses.

    Responses: {json.dumps(api_results)}

    - Ensure all necessary details are included.
    - If any essential data is missing, indicate what additional API calls are required.
    - Remove redundant or irrelevant information.
    - Format the response in a human-readable way.
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": validation_prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# 🔹 FINAL RESPONSE GENERATION
def generate_final_response(validated_response):
    """Return the final structured response after validation."""
    
    if validated_response["status"] == "complete":
        return validated_response["response"]
    
    elif validated_response["status"] == "incomplete":
        missing_data = validated_response["missing_data"]
        return f"⚠️ Some data is missing: {', '.join(missing_data)}. Please try again."

    return "❌ Unable to generate a valid response."

# 🔹 MAIN PROCESSING FUNCTION
def handle_user_query(user_input, user_id="default"):
    """Handle the full user query lifecycle from analysis to response validation."""
    
    structured_queries = analyze_query(user_input)
    api_results = execute_api_calls(structured_queries)
    validated_response = validate_and_refine_response(api_results)

    store_session(user_id, user_input, validated_response["response"])

    return generate_final_response(validated_response)

# 🔹 INTERACTIVE CHATBOT LOOP
def main():
    print("🎬 TMDB Chatbot: Ask me about movies, actors, or trending films!")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 🎬")
            break

        response = handle_user_query(user_input)
        print("\n", response)

if __name__ == "__main__":
    main()
