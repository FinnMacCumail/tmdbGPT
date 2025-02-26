import openai
import json
import requests
import os
from dotenv import load_dotenv

# Load API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3/"

# OpenAI API Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🔹 Define Generalized TMDB API Function
tmdb_functions = [
    {
        "type": "function",
        "function": {
            "name": "tmdb_api_request",
            "description": (
                "Fetch data from TMDB API dynamically based on user intent. "
                "Ensure correct API parameters (like 'query' for searches) are included. "
                "For 'search' endpoints, 'params' must always include a 'query' field with a relevant value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "The TMDB API endpoint (e.g., 'search/person', 'movie/popular')."
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Query parameters for the API request. "
                            "For 'search/person', 'params' must include {'query': 'Actor Name'}. "
                            "For 'movie/popular', 'params' may be an empty object {}."
                        ),
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for TMDB, required for 'search' endpoints."
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": True
                    }
                },
                "required": ["endpoint", "params"]
            }
        }
    }
]

# 🔹 Flexible TMDB API Request Function
def call_tmdb_api(endpoint, params=None):
    """Make a flexible request to the TMDB API with improved error handling."""
    
    url = f"{TMDB_BASE_URL}{endpoint}?api_key={TMDB_API_KEY}"

    if params:
        for key, value in params.items():
            url += f"&{key}={value}"

    response = requests.get(url).json()

    print(f"🔍 Debug: TMDB API Response: {response}")

    # Handle TMDB API errors
    if "status_code" in response and response["status_code"] != 200:
        return {"error": f"❌ TMDB API error: {response.get('status_message', 'Unknown error')}"}    

    return response

# 🔹 OpenAI Tool Calling for Dynamic API Execution
def handle_user_query(user_input, retries=3):
    """Use OpenAI tool calling to dynamically construct and execute API requests, ensuring valid parameters."""
    
    if retries == 0:
        return {"error": "❌ OpenAI failed to provide correct API parameters after multiple attempts."}

    # Force OpenAI to always include the correct 'params' structure in its response
    system_message = """
    You are a movie assistant that constructs correct TMDB API requests.

    - When calling 'search' endpoints, always include a 'params' field with a 'query' key.
    - Example of a correct API call:
        {
            "endpoint": "search/person",
            "params": { "query": "Sofia Coppola" }
        }
    - If the endpoint does not require parameters, use an empty dictionary ({}).
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
        tools=tmdb_functions,
        tool_choice="auto"
    )

    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        parameters = json.loads(tool_call.function.arguments)

        # Validate API request structure
        if "params" not in parameters or ("query" not in parameters["params"] and parameters["endpoint"].startswith("search/")):
            print(f"⚠️ Warning: 'params' missing or incorrect. Retrying {retries-1} more times...")
            return handle_user_query(user_input, retries - 1)  # Retry with reduced attempts

        # Debugging print
        print(f"🛠️ Debug: API Call - Endpoint: {parameters['endpoint']}, Params: {parameters['params']}")

        # Make the API request with corrected parameters
        return call_tmdb_api(parameters["endpoint"], parameters["params"])

    return {"error": "❌ No valid tool call was made."}


# 🔹 Validate & Refine API Response Using OpenAI
def validate_and_refine_response(api_results):
    """Use OpenAI to validate and refine the API response dynamically."""

    validation_prompt = f"""
    Validate and refine the following TMDB API responses.

    API Results: {json.dumps(api_results)}

    - Ensure all necessary details are included.
    - If essential data is missing, indicate what additional API calls are required.
    - Format the response in a human-readable way.

    Example Output:
    {{
        "status": "complete",
        "response": "Sofia Coppola has directed 7 films: The Virgin Suicides (1999), Lost in Translation (2003)..."
    }}

    OR

    {{
        "status": "incomplete",
        "missing_data": ["movie_credits"],
        "response": "Some information is missing. Please fetch movie credits for Sofia Coppola."
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": validation_prompt}],
            temperature=0
        )

        # Ensure response exists
        if not response or not response.choices:
            return {"error": "❌ OpenAI returned an empty validation response."}

        return json.loads(response.choices[0].message.content)

    except json.JSONDecodeError:
        return {"error": "❌ OpenAI response could not be parsed as JSON."}

    except Exception as e:
        return {"error": f"❌ OpenAI API error: {str(e)}"}
    
def validate_and_generate_response(user_query, api_results):
    """Ensure OpenAI generates a response based ONLY on TMDB API results."""

    # Ensure API returned valid data
    if "results" in api_results and api_results["results"]:
        relevant_data = api_results["results"][0]  # Take the first match
    else:
        return "❌ No TMDB data found. Try a different query."

    # Extract meaningful fields
    person_name = relevant_data.get("name", "Unknown")
    known_for = relevant_data.get("known_for", [])

    # Format known movies
    movie_list = [
        f'"{movie["title"]}" ({movie.get("release_date", "Unknown Year")})'
        for movie in known_for
    ]

    # Ensure OpenAI uses only valid TMDB data
    prompt = f"""
    You are a movie assistant using real TMDB data.

    - **User Query**: "{user_query}"
    - **Extracted API Data**:
        - Name: {person_name}
        - Known for: {", ".join(movie_list)}

    **Instructions:**
    1. Generate a response using ONLY the provided TMDB data.
    2. DO NOT invent movies or details not present in the data.
    3. Format the response in a natural, conversational style.
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()



def validate_openai_api_request(parameters):
    """Ensure OpenAI provided a valid API request format with required parameters."""
    
    # Check if 'params' exist for endpoints that require them
    if "params" not in parameters:
        print(f"⚠️ Warning: 'params' missing. Asking OpenAI to regenerate API request...")
        return False  # Mark request as invalid

    # Check if key fields like 'query' are missing for search-based endpoints
    if parameters["endpoint"].startswith("search/") and "query" not in parameters["params"]:
        print(f"⚠️ Warning: 'query' missing in params for search endpoint.")
        return False

    return True  # ✅ API request is valid


# 🔹 Generate Final Formatted Response
def generate_final_response(validated_response):
    """Return the final structured response after validation."""
    
    if validated_response["status"] == "complete":
        return validated_response["response"]
    
    elif validated_response["status"] == "incomplete":
        missing_data = validated_response["missing_data"]
        return f"⚠️ Some data is missing: {', '.join(missing_data)}. Please try again."

    return "❌ Unable to generate a valid response."

# 🔹 Main Processing Function
def process_user_request(user_input):
    """Ensure OpenAI correctly formats TMDB data before responding."""
    
    api_results = handle_user_query(user_input)

    # Ensure valid data before passing to OpenAI
    if "error" in api_results:
        return api_results["error"]

    return validate_and_generate_response(user_input, api_results)


# 🔹 Interactive Chatbot Loop
def main():
    print("🎬 TMDB Chatbot: Ask me about movies, actors, or trending films!")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 🎬")
            break

        response = process_user_request(user_input)
        print("\n", response)

if __name__ == "__main__":
    main()
