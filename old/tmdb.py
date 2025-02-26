import openai
import requests
import json
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage

# TMDB API Configuration
API_KEY="1370b21e6746a8aa1aff4b6e5b9fd92e"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = API_KEY  # 🔹 Replace with your actual TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3/"

# OpenAI API Configuration
#OPENAI_API_KEY = "your_openai_api_key"  # 🔹 Replace with your actual OpenAI API key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🔹 Function to filter relevant OpenAI functions based on user query
def get_relevant_functions(user_input):
    """Filter the function list based on the user's intent."""
    
    # 🔹 Prioritize people searches
    if any(word in user_input.lower() for word in ["who is", "biography", "filmography", "actor", "director"]):
        return [tmdb_dispatcher_function]  # Ensure people queries go to the dispatcher

    # 🔹 Movies and trending-related requests
    if "popular" in user_input or "trending" in user_input or "details" in user_input:
        return [tmdb_dispatcher_function]

    # 🔹 Search & Discovery (fallback)
    if "find me" in user_input or "search" in user_input or "filter" in user_input:
        return [tmdb_dispatcher_function]

    return [tmdb_dispatcher_function]  # Default to the dispatcher


# 🔹 Dispatcher Function for TMDB API Requests
tmdb_dispatcher_function = {
    "type": "function",  # ✅ Fix: Explicitly declare this as a function tool
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
            },
            "movie_id": {
                "type": "integer",
                "description": "The TMDB movie ID (if requesting movie details)."
            },
            "person_id": {
                "type": "integer",
                "description": "The TMDB person ID (if requesting person credits)."
            }
        },
        "required": ["category", "action"]
    }
}

# 🔹 Function to interact with OpenAI and determine action
def get_openai_response(user_input):
    """Send user query to OpenAI and determine appropriate TMDB API call."""
    
    functions = get_relevant_functions(user_input)  # Get relevant functions dynamically

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": user_input}],
        tools=[{"type": "function", "function": tmdb_dispatcher_function}],  # ✅ Correct format
        tool_choice="auto"
    )


    print("🛠 Debug: OpenAI Function Call →", response.choices[0].message.tool_calls)  # ✅ ADD DEBUGGING

    return response

# 🔹 Function to make TMDB API Requests
def call_tmdb_api(endpoint, params=None):
    """Make a request to the TMDB API and return the response."""
    
    url = f"{TMDB_BASE_URL}{endpoint}?api_key={TMDB_API_KEY}"
    
    if params:
        for key, value in params.items():
            url += f"&{key}={value}"

    response = requests.get(url)
    
    return response.json()

import json

def process_openai_response(response):
    """Extract function call from OpenAI response and make the appropriate TMDB API call."""
    
    if response.choices and response.choices[0].message.tool_calls:
        tool_calls = response.choices[0].message.tool_calls
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            parameters = json.loads(tool_call.function.arguments)

            category = parameters.get("category")
            action = parameters.get("action")
            query = parameters.get("query", None)

            # 🔹 Correct OpenAI's function call if it misclassifies a person search
            if category == "search" and query:
                print(f"⚠️ OpenAI misclassified a person search. Fixing query for {query}...")
                return search_tmdb_person(query)  # Redirect to person search

            # 🔹 Handle correctly classified people queries
            if category == "people" and action == "search" and query:
                return search_tmdb_person(query)

    return {"error": "❌ Invalid request"}

import requests

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


# 🔹 Main Function: User Interaction
def main():
    """Main interactive loop for TMDB chatbot."""
    
    print("🎬 TMDB Chatbot: Ask me about movies, actors, or trending films!")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 🎬")
            break

        # Get OpenAI response
        response = get_openai_response(user_input)
        
        # Process OpenAI's function call
        tmdb_data = process_openai_response(response)
        
        # Display formatted results
        if "results" in tmdb_data:
            for item in tmdb_data["results"][:5]:  # Show top 5 results
                title = item.get("title") or item.get("name")
                release_date = item.get("release_date") or item.get("first_air_date", "Unknown Release Date")
                overview = item.get("overview", "No description available.")
                
                print(f"\n🎬 {title} ({release_date})\n📜 {overview}\n")

        elif "error" in tmdb_data:
            print(f"❌ {tmdb_data['error']}")

        else:
            print(f"\n{tmdb_data}\n")

# 🔹 Run the chatbot
if __name__ == "__main__":
    main()
