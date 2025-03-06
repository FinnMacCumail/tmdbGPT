import os
import json
import requests
import logging
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import re

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys
dotenv_path = os.path.join(os.getcwd(), ".env") 
load_dotenv(dotenv_path, override=True)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TMDB_API_KEY:
    logger.error("❌ TMDB_API_KEY is missing or incorrect! Check your .env file.")
    exit(1)  # ✅ Stop execution if API key is missing
else:
    logger.info(f"✅ TMDB_API_KEY loaded: {TMDB_API_KEY}")  # ✅ Correct

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY is missing! Check your .env file.")



# Connect to ChromaDB
CHROMA_DB_PATH = "embeddings/chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection("tmdb_queries")


class ResponseAgent:
    """Formats responses using OpenAI's LLM."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate_response(self, state, user_query):
        """Formats the response dynamically using LLM."""
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
    """Manages query execution and routes tasks to specialized agents."""

    def __init__(self):
        self.state = {}

    def handle_query(self, user_query):
        """Processes the user query dynamically."""
        logger.info(f"🤖 Handling query: {user_query}")

        steps = PlannerAgent().generate_plan(user_query)
        if not steps:
            return "❌ Unable to process your request."

        self.state["steps"] = steps
        self.state["results"] = {}
        self.state["user_query"] = user_query

        for step in steps:
            logger.info(f"🔍 Executing step: {step}")
            ExecutionAgent().execute_step(step, self.state)

        return ResponseAgent().generate_response(self.state, user_query)


class PlannerAgent:
    """Uses LLM to break down user queries into structured steps."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def extract_entity(self, user_query):
        """
        Extracts the relevant entity (person name, movie title, etc.) from a user query.

        Example:
        - "Who is Sofia Coppola?" → "Sofia Coppola"
        - "Tell me about Lost in Translation." → "Lost in Translation"
        """
        function_call_schema = {
            "name": "extract_entity",
            "description": "Extract the relevant entity (person name, movie title, TV show) from a user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The extracted entity needed for the API request."}
                },
                "required": ["entity"]
            }
        }

        prompt = f"""
        Extract the most relevant entity from the following user query:
        "{user_query}"

        - If the query is about a person, return only the person's name.
        - If the query is about a movie, return only the movie title.
        - If the query is about a TV show, return only the TV show title.

        Example outputs:
        - "Who is Sofia Coppola?" → "Sofia Coppola"
        - "Tell me about Lost in Translation." → "Lost in Translation"
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}],
                tools=[{"type": "function", "function": function_call_schema}],
                tool_choice="auto",
                temperature=0  # ✅ Reduce randomness for better accuracy
            )

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                logger.warning("⚠️ OpenAI did not return an extracted entity.")
                return ""

            extracted_data = json.loads(tool_calls[0].function.arguments)
            entity = extracted_data.get("entity", "").strip()

            if not entity:
                logger.warning("⚠️ Extracted entity is empty.")
                return ""

            logger.info(f"✅ Extracted entity from user query: {entity}")

            return entity

        except Exception as e:
            logger.error(f"❌ Failed to extract entity: {e}")
            return ""

    def generate_plan(self, user_query):
        """
        Generates structured steps using OpenAI's LLM and validates API endpoints with ChromaDB.
        """
        extracted_entity = self.extract_entity(user_query)

        if not extracted_entity:
            logger.error("❌ Could not extract a valid entity from user query.")
            return None

        prompt = f"""
        Given the user query: "{user_query}", break it down into logical API calls.
        Identify if the query requires:
        - A single API call (like trending movies).
        - Multiple steps (like finding a person first, then getting their movie credits).

        Do NOT return API endpoints. Only return the intent and required parameters.

        Example output:
        {{
            "steps": [
                {{
                    "intent": "find_person",
                    "parameters": {{"query": "Sofia Coppola"}}
                }},
                {{
                    "intent": "get_person_details",
                    "parameters": {{}}
                }}
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2
            )
            plan = json.loads(response.choices[0].message.content)["steps"]
        except Exception as e:
            logger.error(f"❌ LLM Planner failed: {e}")
            return None

        # ✅ Step 2: Determine Correct Endpoints Using ChromaDB
        validated_steps = []
        for step in plan:
            intent = step["intent"]
            stored_query = ChromaDBAgent().search_chroma_db(intent)  # ✅ Query ChromaDB with intent

            if not stored_query:
                logger.warning(f"❌ No API mapping found in ChromaDB for intent: {intent}")
                continue

            if isinstance(stored_query, list) and stored_query:
                stored_query = stored_query[0]

            if "solution" not in stored_query:
                logger.error(f"❌ Retrieved query does not contain 'solution' key: {stored_query}")
                continue

            try:
                stored_query = json.loads(stored_query["solution"])
            except json.JSONDecodeError:
                logger.error(f"❌ Failed to parse stored query JSON: {stored_query}")
                continue

            # ✅ Replace the endpoint in the step with the correct ChromaDB result
            step["endpoint"] = stored_query.get("endpoint")
            validated_steps.append(step)

        return validated_steps if validated_steps else None


class ChromaDBAgent:
    """Agent for searching API mappings in ChromaDB."""

    def search_chroma_db(self, query):
        """Finds the correct API call mapping from ChromaDB."""
        search_results = collection.query(query_texts=[query], n_results=1)
        return search_results["metadatas"][0] if search_results and search_results["ids"] else None


class ExecutionAgent:
    """Handles API calls dynamically and resolves placeholders."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def extract_placeholders(self, endpoint):
        """Extracts placeholders from API endpoints like `{person_id}`."""
        return re.findall(r'\{(.*?)\}', endpoint)

    def resolve_placeholders(self, endpoint, state):
        """Replaces placeholders with extracted values from state."""
        placeholders = self.extract_placeholders(endpoint)
        missing_placeholders = []

        for placeholder in placeholders:
            if placeholder in state.get("values", {}):
                value = state["values"][placeholder]
                endpoint = endpoint.replace(f"{{{placeholder}}}", str(value))
            else:
                missing_placeholders.append(placeholder)

        return endpoint, missing_placeholders

    def execute_step(self, step, state):
        """Executes an API call dynamically and ensures the correct API mapping is retrieved."""
        logger.info(f"🔍 Executing step: {step}")

        # ✅ Determine whether this step is a primary search step or a secondary API call
        if step["endpoint"].startswith("/search/"):  # ✅ Primary search API
            search_key = state.get("user_query", "")
        else:  # ✅ Secondary API step that depends on extracted data
            search_key = step["endpoint"]

        if not search_key:
            logger.error("❌ Missing search key for ChromaDB lookup.")
            return

        stored_query = ChromaDBAgent().search_chroma_db(search_key)

        if not stored_query:
            logger.warning(f"❌ No API mapping found in ChromaDB for: {search_key}")
            return

        if isinstance(stored_query, list) and stored_query:
            stored_query = stored_query[0]  

        if "solution" not in stored_query:
            logger.error(f"❌ Retrieved query does not contain 'solution' key: {stored_query}")
            return

        try:
            stored_query = json.loads(stored_query["solution"])
        except json.JSONDecodeError:
            logger.error(f"❌ Failed to parse stored query JSON: {stored_query}")
            return

        endpoint = stored_query.get("endpoint")
        params = stored_query.get("parameters", step.get("parameters", {}))

        resolved_endpoint, missing_placeholders = self.resolve_placeholders(endpoint, state)

        # ✅ Ensure required placeholders are available before executing dependent steps
        if missing_placeholders:
            logger.warning(f"⚠️ Missing placeholders: {missing_placeholders}. Trying extraction from previous response.")
            last_response = state.get("last_response", {})
            extracted_data = LLMAgent().extract_relevant_data(step, last_response)

            if extracted_data:
                state.setdefault("values", {}).update(extracted_data)
                resolved_endpoint, remaining_missing = self.resolve_placeholders(endpoint, state)

                if remaining_missing:
                    logger.error(f"❌ Still missing placeholders after extraction: {remaining_missing}")
                    return  # Stop execution if placeholders are still missing
            else:
                logger.error(f"❌ Placeholder extraction failed. Cannot continue.")
                return  # Stop execution if we couldn't extract anything

        logger.info(f"🔍 Final API Call: {resolved_endpoint}")
        result = self.execute_tmdb_api(resolved_endpoint, params)

        if result:
            state["last_response"] = result
            extracted_data = LLMAgent().extract_relevant_data(step, result)
            state.setdefault("values", {}).update(extracted_data)




    def execute_tmdb_api(self, endpoint, params):
        """Calls the TMDB API dynamically with correct authentication."""
        base_url = "https://api.themoviedb.org/3"

        if not TMDB_API_KEY:
            logger.error("❌ TMDB_API_KEY is missing. Check your .env file.")
            return None

        # ✅ Ensure "query" is a raw string, NOT manually encoded
        if "query" in params and isinstance(params["query"], str):
            logger.info(f"🔍 Raw Query Parameter Before Sending: {params['query']}")

        request_url = f"{base_url}{endpoint}"
        logger.info(f"🔍 Sending request to: {request_url} with params {params}")

        headers = {
            "Authorization": f"Bearer {TMDB_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.get(request_url, params=params, headers=headers)
            response_data = response.json()

            # ✅ Log full API response
            logger.info(f"✅ API Response for {endpoint}: {json.dumps(response_data, indent=2)}")

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
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def extract_relevant_data(self, step, api_response):
        """
        Extracts necessary values from an API response based on required placeholders.
        
        - Identifies what needs to be extracted based on the next API call.
        - Uses OpenAI function calling for structured, reliable extraction.

        :param step: The current step in the execution plan.
        :param api_response: The raw JSON response from TMDB API.
        :return: Dictionary containing extracted values.
        """

        if not api_response or "results" not in api_response or not api_response["results"]:
            logger.error("❌ API response is empty or malformed. Cannot extract data.")
            return {}

        # ✅ Identify required placeholders from the next step
        placeholders_needed = ExecutionAgent().extract_placeholders(step["endpoint"])

        if not placeholders_needed:
            logger.info("✅ No placeholders needed for this step.")
            return {}

        # ✅ Convert API response to JSON string to avoid formatting issues
        formatted_json = json.dumps(api_response, indent=2)

        # ✅ Define OpenAI function calling schema
        function_call_schema = {
            "name": "extract_values",
            "description": "Extract necessary values from API response based on placeholders.",
            "parameters": {
                "type": "object",
                "properties": {placeholder: {"type": "string"} for placeholder in placeholders_needed},
                "required": placeholders_needed
            }
        }

        # ✅ Construct prompt for OpenAI
        prompt = f"""
        Given the following API response:
        {formatted_json}

        - Identify and extract only the necessary values for the placeholders: {placeholders_needed}.
        - Ensure the extracted values match their correct keys.

        Return the extracted values as a JSON object.
        """

        try:
            # ✅ Use OpenAI function calling
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}],
                tools=[{"type": "function", "function": function_call_schema}],
                tool_choice="auto",
                temperature=0.2
            )

            # ✅ Extract structured JSON output from OpenAI response
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                logger.warning(f"⚠️ Warning: OpenAI returned no tool calls for extraction.")
                return {}

            extracted_data = json.loads(tool_calls[0].function.arguments)

            logger.info(f"✅ Extracted values from API response: {extracted_data}")

            return extracted_data

        except Exception as e:
            logger.error(f"❌ LLM Extraction failed: {e}")
            return {}


def main():
    """Interactive chatbot loop for querying TMDB."""
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print(OrchestratorAgent().handle_query(user_input))


if __name__ == "__main__":
    main()
