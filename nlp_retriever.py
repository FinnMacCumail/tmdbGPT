import json
import chromadb
import requests
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import ast  # Import to parse JSON-like strings
from dotenv import load_dotenv


# Load API keys
dotenv_path = os.path.join(os.getcwd(), ".env") 
load_dotenv(dotenv_path, override=True)

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

def match_query_to_cluster(query, n_results=3):
    """Find the closest N clusters to the user query using cosine similarity."""
    query_vector = model.encode([query]).tolist()
    search_results = collection.query(query_texts=[query], n_results=n_results)

    # ✅ Debugging: Print multiple search results
    #print("🛠️ Debug: ChromaDB search results:", search_results)

    if not search_results or "metadatas" not in search_results or not search_results["metadatas"]:
        print("❌ No matching clusters found in ChromaDB.")
        return []

    return search_results["metadatas"]  # Return multiple clusters instead of one

def generate_openai_function_call(user_query, matched_clusters):
    """Use OpenAI function calling to determine the correct API call from multiple clusters."""
    if not matched_clusters:
        print("❌ No cluster data available for OpenAI.")
        return []

    function_schemas = []
    
    for i, cluster_data in enumerate(matched_clusters[0]):  # Extract first element (list of metadata)
        function_schemas.append({
            "type": "function",
            "function": {
                "name": f"select_api_function_{i}",
                "description": f"Select the best API function for: {cluster_data['description']}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint": {"type": "string"},
                        "method": {"type": "string"},
                        "parameters": {"type": "object"}
                    },
                    "required": ["endpoint", "method"]
                }
            }
        })

    user_prompt = f"""
    **User Query**: "{user_query.strip().lower()}"  # ✅ Ensure proper formatting

    **Available API Functions**:
    {json.dumps(matched_clusters[0], indent=2)}

    **Task**:
    - Select the most relevant API function for the given user query.
    - If the query is about a **person (e.g., actor, director)**, choose **`/search/person`**.
    - If the query is about a **movie**, choose **`/search/movie`**.
    - If multiple API calls are required, return all of them.

    **Example Response for a multi-step query**:
    {{
      "functions": [
        {{
          "endpoint": "/search/person",
          "method": "GET",
          "parameters": {{
            "query": "sofia coppola"
          }}
        }}
      ]
    }}

    **Return only a valid JSON response, do not explain your choice.**
    """

    #print("🛠️ Debug: OpenAI User Prompt Sent:\n", user_prompt)

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": user_prompt}],
        tools=function_schemas,
        tool_choice="auto",
        response_format={"type": "json_object"}
    )

    #print("🛠️ Debug: Raw OpenAI Response:\n", response)

    if response.choices and response.choices[0].message.tool_calls:
        try:
            tool_call = response.choices[0].message.tool_calls[0]
            return [json.loads(tool_call.function.arguments)]  # Extract JSON function call
        except json.JSONDecodeError:
            print("❌ Failed to parse OpenAI tool call output.")
            return []

    if response.choices and response.choices[0].message.content:
        try:
            return json.loads(response.choices[0].message.content)["functions"]
        except json.JSONDecodeError:
            print("❌ Failed to parse OpenAI function output.")
            return []
    
    print("❌ OpenAI did not return a valid function call.")
    return []

def execute_api_call(api_function):
    """Execute the API call and return the response."""
    url = f"https://api.themoviedb.org/3{api_function['endpoint']}"
    headers = {"Authorization": f"Bearer {TMDB_API_KEY}", "Content-Type": "application/json"}

    # ✅ Print request details for debugging
    print(f"🔍 Debug: Making API Call to {url}")
    print(f"🔍 Debug: Request Parameters: {api_function.get('parameters', {})}")
    
    response = requests.request(api_function["method"], url, headers=headers, params=api_function.get("parameters", {}))

    # ✅ Print raw API response
    #print("🔍 Debug: Raw API Response:", response.json())

    return response.json()

def summarize_response(api_response):
    """Summarize API response using OpenAI's latest API."""
    
    # ✅ Check if API response contains an error
    if not api_response or "status_code" in api_response or "results" in api_response and len(api_response["results"]) == 0:
        return "❌ No results found for the current query. Try a different search term."

    client = openai.OpenAI(api_key=OPENAI_API_KEY)  # ✅ Correct client initialization
    
    prompt = f"Summarize the following API response in natural language: {json.dumps(api_response)}"
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

def main():
    user_query = input("Enter your query: ")
    print("🔍 Matching query to closest API cluster...")
    matched_clusters = match_query_to_cluster(user_query)

    if not matched_clusters:
        print("❌ No relevant API function found.")
        return

    print("🤖 Selecting best API function using OpenAI...")
    api_function = generate_openai_function_call(user_query, matched_clusters)

    #print("🛠️ Debug: OpenAI function output:", api_function)

    # ✅ Fix: Ensure API function output is valid
    if not api_function or not isinstance(api_function, list) or not api_function[0].get("endpoint"):
        print("❌ OpenAI did not return a valid API function.")
        return

    print("🌐 Executing API call...")
    response = execute_api_call(api_function[0])  # ✅ Pass first function

    print("📝 Summarizing API response...")
    summary = summarize_response(response)

    print("✅ API Response:")
    print(summary)

if __name__ == "__main__":
    main()
