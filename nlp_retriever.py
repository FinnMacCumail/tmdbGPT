import json
import chromadb
import requests
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import ast  # Import to parse JSON-like strings
from dotenv import load_dotenv
import spacy


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

class IntentAnalyzer:
    """
    Analyzes a user query by extracting entities, applying domain-specific rules to map those entities 
    into intended API calls, and, if needed, falls back on an LLM to refine the interpretation.
    
    Attributes:
      nlp: A spaCy NLP model instance for entity extraction.
      llm_client: An object with a 'generate_response(prompt)' method for LLM-assisted refinement.
      entity_mapping: A dict mapping spaCy entity labels to domain-specific keys.
      rule_functions: A list of functions that implement domain-specific rules for intent detection.
    """
    
    def __init__(self, nlp_model=None, llm_client=None, entity_mapping=None, rule_functions=None):
        # Load a spaCy model (default to a lightweight model) if one is not provided.
        self.nlp = nlp_model if nlp_model is not None else spacy.load("en_core_web_trf")
        self.llm_client = llm_client  # Expected to have a generate_response(prompt) method.
        
        # Allow entity mapping to be injected; provide a default mapping.
        self.entity_mapping = entity_mapping if entity_mapping is not None else {
            "PERSON": "person", 
            "WORK_OF_ART": "movie_title",
            "DATE": "date",
            "GPE": "region",
            "ORG": "organization",
            "EVENT": "event",
            "NORP": "group"
        }
        
        # Domain-specific rule functions can be injected to avoid hardcoding.
        # Each function should accept (query, entities) and return a dictionary of intent mappings.
        self.rule_functions = rule_functions if rule_functions is not None else [
            self.rule_search_person,
            self.rule_search_movie,
            self.rule_filter_by_date
        ]
    
    def extract_entities(self, query):
        """
        Uses the spaCy model to extract entities from the query and maps them to domain-specific keys.
        """
        doc = self.nlp(query)
        extracted = {}
        for ent in doc.ents:
            # Map the spaCy entity label using the injected mapping.
            key = self.entity_mapping.get(ent.label_, ent.label_)
            # Concatenate if the same key appears more than once.
            if key in extracted:
                extracted[key] += f" {ent.text}"
            else:
                extracted[key] = ent.text
        return extracted
    
    def rule_search_person(self, query, entities):
        """
        Rule to set an intent for searching a person.
        If a 'person' entity is present or if keywords like 'actor' or 'director' are found,
        set the 'search_person' intent with a query parameter.
        """
        intents = {}
        if "person" in entities:
            intents["search_person"] = {"query": entities["person"]}
        elif any(term in query.lower() for term in ["actor", "director", "celebrity"]):
            intents["search_person"] = {"query": query}
        return intents
    
    def rule_search_movie(self, query, entities):
        """
        Rule to set an intent for searching a movie.
        If a movie title or work of art is detected or if the query contains movie/film related keywords,
        set the 'search_movie' intent.
        """
        intents = {}
        if "movie_title" in entities:
            intents["search_movie"] = {"query": entities["movie_title"]}
        elif any(term in query.lower() for term in ["movie", "film"]):
            intents["search_movie"] = {"query": query}
        return intents
    
    def rule_filter_by_date(self, query, entities):
        """
        Rule to capture date filters. If a date entity is detected, assign it to a filter intent.
        """
        intents = {}
        if "date" in entities:
            intents["filter_date"] = {"date": entities["date"]}
        return intents
    
    def basic_intent_detection(self, query, entities):
        """
        Runs all the configured rule functions and aggregates their outputs.
        """
        intents = {}
        for rule in self.rule_functions:
            result = rule(query, entities)
            # Merge the result dictionaries.
            intents.update(result)
        return intents
    
    def refine_intent_with_llm(self, query, entities, fallback_plan=None):
        """
        If basic intent detection yields insufficient results, use an LLM to generate a refined plan.
        Expects the llm_client to return a valid JSON string.
        """
        if self.llm_client is None:
            return fallback_plan if fallback_plan is not None else {}
        
        prompt = (
            f"Decompose the following query into a JSON structure mapping intents to parameters.\n"
            f"Query: \"{query}\"\n"
            f"Recognized entities: {json.dumps(entities)}\n"
            "Return a JSON object with a 'steps' key that is a list of intent dictionaries."
        )
        response = self.llm_client.generate_response(prompt)
        try:
            plan = json.loads(response)
            return plan.get("steps", {})
        except Exception as e:
            print("LLM refinement error:", e)
            return fallback_plan if fallback_plan is not None else {}
    
    def analyze(self, query):
        """
        Runs the full analysis pipeline:
          1. Extract entities.
          2. Apply domain-specific rules to derive basic intents.
          3. Optionally, if basic intents are ambiguous or missing, refine via an LLM.
          
        Returns a dictionary with the extracted entities and either a set of basic intents 
        or a refined multi-step plan.
        """
        entities = self.extract_entities(query)
        intents = self.basic_intent_detection(query, entities)
        if not intents:
            # Use LLM fallback if no clear intents are determined.
            refined = self.refine_intent_with_llm(query, entities)
            return {"entities": entities, "steps": refined}
        return {"entities": entities, "intents": intents}

import openai

import openai

from openai import OpenAI

class OpenAILLMClient:
    def __init__(self, api_key, model="gpt-4-turbo", temperature=0.2):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, prompt, additional_messages=None, tools=None, tool_choice=None, temperature=None):
        if temperature is None:
            temperature = self.temperature

        messages = [{"role": "system", "content": prompt}]
        if additional_messages:
            messages.extend(additional_messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature
        )

        message = response.choices[0].message
        if message.tool_calls:
            return message.tool_calls[0].function.arguments
        else:
            return message.content.strip()
 
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

def generate_openai_function_call(user_query, matched_clusters, llm_client):
    """Use LLM function calling (via llm_client) to determine the correct API call from multiple clusters."""
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
    **User Query**: "{user_query.strip().lower()}"  # Ensure proper formatting

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

    # Use our OpenAILLMClient instance (llm_client) to generate the response.
    try:
        response_str = llm_client.generate_response(
            prompt=user_prompt,
            tools=function_schemas,
            tool_choice="auto",
            temperature=0
        )
        # First try to parse a function call output
        try:
            result = [json.loads(response_str)]
            return result
        except json.JSONDecodeError:
            # Fallback: try to parse as a JSON object with a "functions" key.
            parsed = json.loads(response_str)
            return parsed.get("functions", [])
    except Exception as e:
        print("❌ Error in generating function call:", e)
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
    # Instantiate the analyzer with optional parameters.
    llm_client = OpenAILLMClient(api_key=OPENAI_API_KEY)
    intent_analyzer = IntentAnalyzer(llm_client=llm_client)
    
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        # Analyze the query
        analysis_result = intent_analyzer.analyze(user_query)
        
        # Debug print to see what was extracted
        print("Extracted Entities:", analysis_result.get("entities"))
        print("Detected Intents / Steps:", analysis_result.get("intents") or analysis_result.get("steps"))
        
        # Depending on the structure, you might have:
        if "steps" in analysis_result:
            # For multi-step queries, pass the steps to your planner/execution pipeline
            plan = analysis_result["steps"]
            # Example: for step in plan: execute_api_call(step)
        else:
            # For simple queries, use the basic intents directly
            intents = analysis_result.get("intents", {})
            # Example: map the 'search_movie' intent to an API call.
        
        # Continue with the rest of your pipeline (e.g., generating and executing API calls)
        # ...

if __name__ == "__main__":
    main()
