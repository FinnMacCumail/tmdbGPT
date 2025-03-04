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
    """Executes API calls dynamically, ensuring placeholders are replaced using extracted values."""
    
    def execute_step(self, step, state):
        """Executes an API call dynamically and updates the state with extracted values."""

        # Retrieve API mapping from ChromaDB
        stored_query = ChromaDBAgent().search_chroma_db(step["endpoint"])
        logger.info(f"🔍 Retrieved ChromaDB query: {stored_query}")

        if not stored_query:
            logger.warning(f"❌ No API mapping found for {step['endpoint']}")
            return

        if isinstance(stored_query, list) and stored_query:
            stored_query = stored_query[0]

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

        # ✅ Ensure search query parameter is populated ONLY for search endpoints
        if "/search/" in endpoint:
            if "query" in params and not params["query"]:
                params["query"] = step["parameters"].get("query", "")

            if not params.get("query"):
                logger.error("❌ Search query is missing. Cannot proceed with search API call.")
                return  # Prevents an empty request

        # ✅ Inject extracted `person_id` dynamically (no hardcoding)
        extracted_values = state.get("values", {})
        for key, value in extracted_values.items():
            if isinstance(value, (str, int)):  
                endpoint = endpoint.replace(f"{{{key}}}", str(value))

        logger.info(f"🔍 Final API Call with Resolved Placeholders: {endpoint}")

        # ✅ Execute the API call
        result = self.execute_tmdb_api(endpoint, params)

        if not result:
            logger.error(f"❌ API call failed or returned empty response for {endpoint}. Skipping LLM extraction.")
            return

        # ✅ Log the raw API response for debugging
        logger.info(f"✅ Raw API Response for {endpoint}: {json.dumps(result, indent=2)}")

        # ✅ Extract new values from API response using LLM
        extracted_data = LLMAgent().extract_relevant_data(step, result)

        if not extracted_data.get("values"):
            logger.warning("⚠️ No new extracted values from API response. Proceeding with existing data.")

        # ✅ Update state with extracted values dynamically
        state.update(extracted_data.get("values", {}))

        # ✅ Apply extracted `person_id` or other placeholders for next steps
        if "updated_endpoint" in extracted_data and extracted_data["updated_endpoint"]:
            endpoint = extracted_data["updated_endpoint"]

        for key, value in state.get("values", {}).items():
            if isinstance(value, (str, int)):  
                endpoint = endpoint.replace(f"{{{key}}}", str(value))

        logger.info(f"🔍 Final API Call after Extraction: {endpoint}")

        # ✅ Execute the updated API call if needed
        final_result = self.execute_tmdb_api(endpoint, params)
        state["results"][step["intent"]] = final_result


    def execute_tmdb_api(self, endpoint, params):
        """Calls the TMDB API dynamically."""
        base_url = "https://api.themoviedb.org/3"

        if not TMDB_API_KEY:
            logger.error("❌ TMDB_API_KEY is missing. Check your .env file.")
            return None

        # ✅ Add API Key to request parameters
        params["api_key"] = TMDB_API_KEY

        request_url = f"{base_url}{endpoint}"
        logger.info(f"🔍 Sending request to: {request_url} with params {params}")

        try:
            response = requests.get(request_url, params=params)
            response_data = response.json()

            if response.status_code == 200:
                return response_data
            else:
                logger.error(f"❌ API call failed: {request_url} (Status: {response.status_code})")
                logger.error(f"❌ Response: {response_data}")
                return None
        except requests.RequestException as e:
            logger.error(f"❌ Request error: {e}")
            return None

class LLMAgent:
    """Uses LLM reasoning to dynamically extract key values and replace placeholders."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract_relevant_data(self, step, api_response):
        """Extracts dynamic values from API responses and replaces placeholders."""

        if not api_response or "results" not in api_response or not api_response["results"]:
            logger.error("❌ API response is empty or malformed. Cannot extract data.")
            return {}

        # ✅ Convert JSON first (avoiding formatting issues)
        formatted_json = json.dumps(api_response, indent=2)

        # ✅ Use an f-string instead of str.format()
        prompt = f"""
        Given the API response:
        {formatted_json}

        - Identify if the response contains a `person_id` for a search query.
        - If applicable, return "values": {{"person_id": extracted_id}}.
        - If the response does not contain valid results, return {{"values": {{}}}}.

        Only return a JSON response.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2
            )

            raw_output = response.choices[0].message.content.strip()
            logger.info(f"✅ Raw LLM Response: {raw_output}")

            extracted_data = json.loads(raw_output)

            if not isinstance(extracted_data, dict) or "values" not in extracted_data:
                logger.warning("⚠️ LLM returned unexpected structure.")
                return {}

            return extracted_data
        except json.JSONDecodeError:
            logger.error("❌ LLM returned invalid JSON.")
            return {}
        except Exception as e:
            logger.error(f"❌ LLM Extraction failed: {e}")
            return {}


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
