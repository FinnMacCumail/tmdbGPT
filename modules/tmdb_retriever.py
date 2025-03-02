import openai
import os
import json
import chromadb
from dotenv import load_dotenv
import sys

# Add project root to PYTHONPATH dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.endMap import handle_tmdb_dispatcher
from utils.logger import get_logger
from utils.tmdb_functions import load_tmdb_schema

# Initialize logger
logger = get_logger("tmdb_retriever")

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize ChromaDB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "..", "embeddings", "chroma_db")  # ✅ Matches tmdb_embedder.py
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

def query_chromadb(query_text):
    """Search ChromaDB and return best match."""
    logger.debug(f"🔍 Searching ChromaDB for: {query_text}")
    search_results = collection.query(query_texts=[query_text], n_results=1)

    if not search_results["metadatas"] or not search_results["metadatas"][0]:
        logger.warning("⚠️ No valid match found in ChromaDB!")
        return None, 0.0  # No valid match

    best_match = search_results["metadatas"][0][0] if isinstance(search_results["metadatas"][0], list) else search_results["metadatas"][0]
    match_score = search_results["distances"][0][0]  # Extract match confidence score

    return json.loads(best_match.get("solution", "{}")), float(match_score)

def handle_user_query(user_input):
    """Determines the correct API call and retrieves data dynamically."""
    
    # Check ChromaDB first
    best_api_call, match_score = query_chromadb(user_input)

    if best_api_call and match_score > 0.7:
        logger.debug(f"✅ Using Cached API Call - Match Score: {match_score}")
        logger.debug(f"🛠 API Call Sent: {best_api_call}")
        return handle_tmdb_dispatcher(best_api_call)

    # If no match found, use OpenAI to generate a query
    system_message = """
    You are a TMDB API assistant. Your task is to generate structured API calls dynamically based on user requests.

    Ensure responses follow the function signature below:
    {
        "endpoint": "<TMDB API endpoint>",
        "method": "<HTTP method>",
        "parameters": { "<parameter_name>": "<parameter_value>" }
    }
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
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_input}],
        tools=tools,
        tool_choice="auto"
    )

    # Ensure a valid API call is generated
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        parameters = json.loads(tool_call.function.arguments)

        logger.debug(f"🛠 Generated API Call: {parameters}")
        return handle_tmdb_dispatcher(parameters)

    return {"error": "❌ No valid tool call was made."}

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
