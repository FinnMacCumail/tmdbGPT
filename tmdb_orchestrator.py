import os
import json
import requests
import logging
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys

dotenv_path = os.path.join(os.getcwd(), ".env") 
load_dotenv(dotenv_path)

# Get API keys from environment
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TMDB_API_KEY:
    print("❌ ERROR: TMDB_API_KEY is missing! Check your .env file.")
else:
    print(f"✅ TMDB_API_KEY loaded: {TMDB_API_KEY[:5]}******")


if not OPENAI_API_KEY:
    logger.error("❌ Missing OPENAI_API_KEY. Ensure it's set in your .env file.")

# Correctly set ChromaDB path to ~/embeddings/chroma_db
CHROMA_DB_PATH = "embeddings/chroma_db"

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection("tmdb_queries")

class ResponseAgent:
    """Uses LLM to dynamically format the final response."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, state, user_query):
        """
        Uses OpenAI's LLM to format the response dynamically.
        """
        prompt = f"""
        Given the following retrieved data:
        {json.dumps(state['results'], indent=2)}

        And the user query: "{user_query}"

        - Identify the most relevant information.
        - Extract key details.
        - Format the response naturally.

        Output only the final response.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ LLM Response Generation failed: {e}")
            return "I found some information, but I'm unable to format it correctly."


class OrchestratorAgent:
    """
    Main agent that manages query execution, keeps track of state,
    and routes tasks to specialized agents.
    """

    def __init__(self):
        self.state = {}

    def handle_query(self, user_query):
        """Processes the user query dynamically."""
        logger.info(f"🤖 Orchestrator handling query: {user_query}")

        # Step 1: Use PlannerAgent to break down the query
        steps = PlannerAgent().generate_plan(user_query)
        if not steps:
            return "❌ Unable to process your request."

        self.state["steps"] = steps
        self.state["results"] = {}

        # Step 2: Execute steps dynamically
        for step in steps:
            logger.info(f"🔍 Executing step: {step}")
            ExecutionAgent().execute_step(step, self.state)

        # Step 3: Format final response
        return ResponseAgent().generate_response(self.state, user_query)


class PlannerAgent:
    """Uses LLM to break down user queries into structured steps."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_plan(self, user_query):
        """
        Generates structured steps using OpenAI's LLM.
        """
        prompt = f"""
        Given the user query: "{user_query}", break it down into logical API calls.
        Identify if the query requires:
        - A single API call (like trending movies).
        - Multiple steps (like finding a person first, then getting their movie credits).

        Return structured JSON with each step.

        Example output:
        {{
            "steps": [
                {{
                    "intent": "fetch_trending_movies",
                    "endpoint": "/trending/movie",
                    "parameters": {{"time_window": "day"}}
                }}
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(  # Use self.client instead of client
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)["steps"]
        except Exception as e:
            logger.error(f"❌ LLM Planner failed: {e}")
            return None

class ChromaDBAgent:
    """Agent responsible for searching API mappings in ChromaDB."""

    def search_chroma_db(self, query):
        """Finds the correct API call for a given step."""
        search_results = collection.query(query_texts=[query], n_results=1)
        return search_results["metadatas"][0] if search_results and search_results["ids"] else None


class ExecutionAgent:
    """Executes API calls based on planned steps."""

    def execute_step(self, step, state):
        """Executes an API call dynamically and updates the state."""

        search_query = "/search/person"
        stored_query = ChromaDBAgent().search_chroma_db(search_query)
        print(f"test ChromaDB Query for {search_query}: {stored_query}")

        stored_query = ChromaDBAgent().search_chroma_db(step["endpoint"])
        
        logger.info(f"🔍 Retrieved ChromaDB query: {stored_query}")  # Print the stored query

        if not stored_query:
            logger.warning(f"❌ No API mapping found for {step['endpoint']}")
            return

        if isinstance(stored_query, list):  # If stored_query is a list, extract the first item
            stored_query = stored_query[0] if stored_query else {}

        endpoint = stored_query.get("endpoint", None)
        if not endpoint:
            logger.warning(f"❌ No valid endpoint found for {step['endpoint']}")
            return

        params = stored_query.get("parameters", step.get("parameters", {}))
        logger.info(f"🔍 Executing API Call: {endpoint} with {params}")

        result = self.execute_tmdb_api(endpoint, params)
        state["results"][step["intent"]] = result


class ExecutionAgent:
    """Executes API calls based on planned steps."""

    def execute_step(self, step, state):
        """Executes an API call dynamically and updates the state."""
        stored_query = ChromaDBAgent().search_chroma_db(step["endpoint"])
        
        logger.info(f"🔍 Retrieved ChromaDB query: {stored_query}")

        if not stored_query:
            logger.warning(f"❌ No API mapping found for {step['endpoint']}")
            return

        if isinstance(stored_query, list) and stored_query:  # Extract first item if it's a list
            stored_query = stored_query[0]

        # Ensure stored_query is parsed correctly
        if "solution" in stored_query:
            try:
                stored_query = json.loads(stored_query["solution"])
            except json.JSONDecodeError:
                logger.error(f"❌ Failed to parse stored query: {stored_query}")
                return

        endpoint = stored_query.get("endpoint")
        if not endpoint:
            logger.warning(f"❌ No valid endpoint found for {step['endpoint']}")
            return

        params = stored_query.get("parameters", step.get("parameters", {}))
        logger.info(f"🔍 Executing API Call: {endpoint} with {params}")

        result = self.execute_tmdb_api(endpoint, params)
        state["results"][step["intent"]] = result

    def execute_tmdb_api(self, endpoint, params):
        """Calls TMDB API dynamically with proper authentication."""
        base_url = "https://api.themoviedb.org/3"

        if not TMDB_API_KEY:
            logger.error("❌ TMDB_API_KEY is missing. Check your .env file.")
            return None

        # ✅ Ensure API key is passed in query parameters
        params["api_key"] = TMDB_API_KEY

        request_url = f"{base_url}{endpoint}"  # Base URL
        logger.info(f"🔍 Sending request to: {request_url} with params {params}")

        response = requests.get(request_url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"❌ API call failed: {request_url} (Status: {response.status_code})")
            logger.error(f"❌ Response: {response.json()}")  # Log full error details
            return None



# ------------------ MAIN CHATBOT LOOP ------------------ #
def main():
    """Interactive chatbot loop for querying TMDB."""
    print("🎬 TMDB Chatbot: Ask me about movies, actors, or trending films!")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 🎬")
            break

        response = OrchestratorAgent().handle_query(user_input)
        print(response)

if __name__ == "__main__":
    main()
