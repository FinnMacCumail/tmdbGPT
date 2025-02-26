import json
import chromadb
import openai
import requests
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "embeddings", "chroma_db")

# ✅ Ensure ChromaDB connects correctly
print(f"🛠️ Debug: Connecting to ChromaDB at {CHROMA_DB_PATH}")

# Initialize ChromaDB client (persistent storage)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name="tmdb_queries")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def determine_media_type(user_query):
    """
    Determines if the user wants movies or TV shows based on query keywords.
    """
    if "tv" in user_query.lower() or "show" in user_query.lower():
        return "tv"
    return "movie"  # Default to movies


def determine_time_window(user_query):
    """
    Determines if the user wants trending data for 'day' or 'week'.
    """
    if "week" in user_query.lower():
        return "week"
    return "day"  # Default to daily trending


def retrieve_best_tmdb_api_call(user_query, top_k=3):
    """
    Retrieves the most relevant TMDB API call from ChromaDB using similarity search.
    """
    print("🛠️ Debug: Checking stored embeddings in ChromaDB...")
    all_items = collection.get(include=["metadatas"])
    print(f"✅ Found {len(all_items['metadatas'])} stored API calls.")

    # Convert user query to an embedding
    query_embedding = model.encode(user_query).tolist()

    # Perform similarity search in ChromaDB
    search_results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Ensure search results contain valid data
    if not search_results or not search_results.get("metadatas") or not search_results.get("distances"):
        return {"error": "❌ No matching TMDB API found."}

    try:
        best_match_metadata = search_results["metadatas"][0][0].get("solution")
        match_score = search_results["distances"][0][0]  # Extract first float value
    except (IndexError, AttributeError, TypeError):
        return {"error": "❌ Retrieved metadata format is incorrect."}

    print(f"🛠️ Debug: Best Match Retrieved -> {best_match_metadata}")
    print(f"🛠️ Debug: Match Score -> {match_score}")

    # Ensure best_match_metadata is a valid JSON string before parsing
    if not best_match_metadata:
        return {"error": "❌ Retrieved API metadata is empty or malformed."}

    try:
        api_call_details = json.loads(best_match_metadata)
    except json.JSONDecodeError:
        return {"error": "❌ Retrieved API metadata is not valid JSON."}

    # 🔥 Fix: Ensure correct media type
    if api_call_details["endpoint"] == "/trending/{media_type}/{time_window}":
        if "movie" in user_query.lower():
            api_call_details["endpoint"] = "/trending/movie/day"
        else:
            api_call_details["endpoint"] = "/trending/tv/day"  # Default to TV

    # If confidence is low, ask OpenAI for further refinement
    if match_score > 0.7:
        print("⚠️ Low confidence in API match. Asking OpenAI to refine the request...")
        refined_call = refine_with_openai(user_query, api_call_details)
        return refined_call

    return api_call_details



def refine_with_openai(user_query, retrieved_api_call):
    """
    Uses OpenAI to refine API call details when retrieval confidence is low.
    """
    prompt = f"""
    The user asked: "{user_query}"

    The closest API call retrieved is:
    {json.dumps(retrieved_api_call, indent=2)}

    Improve the API call so it accurately matches the user intent.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a TMDB API expert."},
                  {"role": "user", "content": prompt}]
    )

    return json.loads(response.choices[0].message.content)


def execute_tmdb_api_call(api_call_details, user_query):
    """
    Constructs the full TMDB API call dynamically and fetches the data.
    """
    base_url = "https://api.themoviedb.org/3"

    # Handle dynamic parameters for "/trending/{media_type}/{time_window}"
    if "/trending/{media_type}/{time_window}" in api_call_details["endpoint"]:
        media_type = determine_media_type(user_query)
        time_window = determine_time_window(user_query)
        endpoint = f"/trending/{media_type}/{time_window}"
    else:
        endpoint = api_call_details["endpoint"]

    # Construct full API URL
    api_url = f"{base_url}{endpoint}?api_key={TMDB_API_KEY}"

    print(f"\n🛠️ Constructed API URL: {endpoint}")
    print(f"🔗 Fetching Data from: {api_url}")

    # Fetch data
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        print("\n📊 API Response:", json.dumps(data, indent=2))
    else:
        print(f"❌ API Request Failed: {response.status_code} - {response.text}")


if __name__ == "__main__":
    while True:
        user_query = input("\n🔎 Enter a movie-related query (or 'exit' to quit): ")
        if user_query.lower() in ["exit", "quit"]:
            break

        best_api_call = retrieve_best_tmdb_api_call(user_query)

        if "error" in best_api_call:
            print(f"❌ Error: {best_api_call['error']}")
        else:
            execute_tmdb_api_call(best_api_call, user_query)
