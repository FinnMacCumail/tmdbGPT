import openai
import os
import json
import chromadb
from dotenv import load_dotenv
import sys
import logging

# Add project root to PYTHONPATH dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.endMap import handle_tmdb_dispatcher
from utils.logger import get_logger
from utils.tmdb_functions import load_tmdb_schema

# Set logging level to INFO (hides DEBUG logs)
logging.basicConfig(level=logging.INFO)

# Suppress verbose logs from third-party libraries
for lib in ["httpcore", "httpx", "urllib3", "chromadb", "openai"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger("tmdb_retriever")
logger.info("🎬 TMDB Chatbot is running...")

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize ChromaDB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "..", "embeddings", "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

CONFIDENCE_THRESHOLD = 0.9  # Ignore low-confidence matches


### **🔍 Query ChromaDB Function (Updated)**
def query_chromadb(query_text, collection):
    """Search ChromaDB and return the best match with confidence filtering."""
    logger.debug(f"🔍 Searching ChromaDB for: {query_text}")

    try:
        search_results = collection.query(query_texts=[query_text], n_results=1)

        if not search_results or "metadatas" not in search_results or not search_results["metadatas"]:
            logger.warning("⚠️ No valid match found in ChromaDB!")
            return None, 0.0

        best_match_metadata = search_results["metadatas"][0]
        if isinstance(best_match_metadata, list):
            best_match_metadata = best_match_metadata[0]

        match_score = float(search_results["distances"][0][0])  # Confidence score

        if "solution" not in best_match_metadata:
            logger.warning("⚠️ Match found, but missing 'solution' key in metadata.")
            return None, match_score

        # Load solution JSON
        best_match = json.loads(best_match_metadata["solution"])

        # Validate API parameters to prevent placeholder queries
        if "parameters" in best_match:
            for key, value in best_match["parameters"].items():
                if "Specify" in str(value) or "Pass a text query" in str(value):
                    logger.warning(f"⚠️ Ignoring cached query with invalid placeholder text: {value}")
                    return None, match_score  # Forces OpenAI to generate a new query

        # Apply confidence threshold to avoid incorrect matches
        if match_score < CONFIDENCE_THRESHOLD:
            logger.warning(f"⚠️ Match score {match_score:.3f} is below confidence threshold ({CONFIDENCE_THRESHOLD}). Ignoring cached result.")
            return None, match_score

        logger.info(f"✅ ChromaDB Match Found: Score = {match_score:.3f}, Endpoint = {best_match.get('endpoint', 'Unknown')}")
        return best_match, match_score

    except Exception as e:
        logger.error(f"❌ Error querying ChromaDB: {e}", exc_info=True)
        return None, 0.0


### **🧠 Handle User Query Function (Updated)**
def handle_user_query(user_input):
    """Determines the correct API call and retrieves data dynamically."""
    
    # 🔹 Step 1: Check ChromaDB for a Cached Query Match
    best_api_call, match_score = query_chromadb(user_input, collection)

    if best_api_call and match_score >= CONFIDENCE_THRESHOLD:
        logger.info(f"✅ Using Cached API Call (Match Score: {match_score:.3f})")
        return handle_tmdb_dispatcher(best_api_call)

    # 🔹 Step 2: If No Match Found, Generate a New API Call Using OpenAI
    system_message = """
    You are a TMDB API assistant. Your task is to generate structured API calls dynamically based on user requests.

    ### Important Instructions:
    - Extract **only** the movie title, actor name, or director name from the user’s request.
    - If the request is about a **movie**, return only its title.
    - If the request is about a **person**, return only their name.
    - Never include phrases like "Tell me about", "Find all movies with", or "Which films were directed by".
    - Ensure API parameters are correctly structured and do not contain placeholder text.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_api_call",
                "description": "Generate a structured TMDB API request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint": {"type": "string", "description": "The TMDB API endpoint"},
                        "method": {"type": "string", "enum": ["GET", "POST"], "description": "The HTTP method"},
                        "parameters": {
                            "type": "object",
                            "description": "Query parameters for the TMDB API request"
                        }
                    },
                    "required": ["endpoint", "method", "parameters"]
                }
            }
        }
    ]

    logger.info(f"🧠 Sending query to OpenAI for: {user_input}")

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_input}],
            tools=tools,
            tool_choice="required"  # Forces OpenAI to use the tool
        )

        # 🔹 Validate OpenAI Response
        if not response.choices or not response.choices[0].message.tool_calls:
            logger.error(f"❌ OpenAI did not return a valid tool call. Full Response: {response}")
            return {"error": "❌ No valid tool call was made."}

        tool_call = response.choices[0].message.tool_calls[0]

        if not tool_call.function.arguments:
            logger.error("❌ OpenAI response is missing function arguments.")
            return {"error": "❌ OpenAI did not return function arguments."}

        parameters = json.loads(tool_call.function.arguments)

        # Ensure API parameters exist
        if not parameters.get("parameters"):
            logger.warning("⚠️ OpenAI response is missing parameters. Falling back to default search query.")
            parameters["parameters"] = {"query": user_input}  

        # Ensure API endpoint starts with '/'
        if not parameters["endpoint"].startswith("/"):
            parameters["endpoint"] = f"/{parameters['endpoint']}"

        logger.info(f"✅ OpenAI Generated API Call: {parameters}")
        return handle_tmdb_dispatcher(parameters)

    except Exception as e:
        logger.error(f"❌ OpenAI API Error: {e}", exc_info=True)
        return {"error": "❌ Failed to generate API call due to an error."}


### **🎬 Main Interactive Chatbot Loop**
def main():
    """Main interactive loop for TMDB chatbot."""
    print("🎬 TMDB Chatbot: Ask me about movies, actors, or trending films!")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 🎬")
            break

        response = handle_user_query(user_input)
        print(response)


if __name__ == "__main__":
    main()
