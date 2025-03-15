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
from openai import OpenAI
import re

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

nlp = spacy.load("en_core_web_trf")

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
        """Extracts named entities from the query using NLP without any hardcoding."""
        doc = nlp(query)
        extracted = {}

        # Map spaCy entity types dynamically based on TMDB API parameter names
        entity_mappings = {
            "PERSON": "query",  # Used for /search/person
            "WORK_OF_ART": "query",  # Used for /search/movie
            "DATE": "date",
            "GPE": "region"
        }

        for ent in doc.ents:
            if ent.label_ in entity_mappings:
                extracted[entity_mappings[ent.label_]] = ent.text

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

class PlannerAgent:
    def __init__(self, llm_client, chroma_collection, intent_analyzer):
        self.llm_client = llm_client
        self.chroma_collection = chroma_collection
        self.intent_analyzer = intent_analyzer

    def generate_plan(self, query, extracted_entities):
        """
        Dynamically builds an execution plan based on detected API intent and extracted entities.
        """
        intent_data = detect_intents(query, self.chroma_collection, self.intent_analyzer)  # ✅ Now correctly passes intent_analyzer

        if not intent_data:
            print("⚠️ No matching API found for:", query)
            return {"plan": []}

        # ✅ Ensure `extracted_entities` is passed to `extract_required_parameters`
        parameters = extract_required_parameters(intent_data["parameters"], extracted_entities)  # ✅ Fix applied

        for param in parameters.keys():
            if param in extracted_entities:
                parameters[param] = extracted_entities[param]  # ✅ Use extracted entities dynamically

        step = {
            "step": 1,
            "type": "api_call",
            "intent": intent_data["intent"],
            "endpoint": intent_data["endpoint"],
            "method": intent_data["method"],
            "parameters": parameters,  # ✅ Inject dynamically extracted parameters
            "depends_on": None
        }

        return {"plan": [step]}


def match_query_to_cluster(query, chroma_collection, extracted_entities, n_results=5):
    """
    Searches ChromaDB for the best matching API clusters dynamically.

    - Uses ChromaDB's similarity search.
    - Re-ranks results based on extracted entities.
    """
    refined_query = refine_embedding_input(query, extracted_entities)
    search_results = chroma_collection.query(query_texts=[refined_query], n_results=n_results)

    if not search_results or "metadatas" not in search_results or not search_results["metadatas"]:
        print("⚠️ No metadata found in ChromaDB results.")
        return []

    metadata_results = search_results["metadatas"][0]
    distances = search_results["distances"][0]

    # ✅ Re-rank dynamically based on extracted entity types
    entity_weighting = {
        "query": "/search/person",  # If a person entity is found, prioritize `/search/person`
        "movie_title": "/search/movie"
    }

    def rank_function(item, score):
        endpoint = item.get("endpoint", "")
        weight = 0

        for entity_type, preferred_endpoint in entity_weighting.items():
            if entity_type in extracted_entities and preferred_endpoint in endpoint:
                weight -= 1  # Boost priority for preferred endpoint

        return score + weight  # Lower score is better

    # ✅ Sort results dynamically based on weighted scores
    sorted_results = sorted(
        zip(metadata_results, distances),
        key=lambda x: rank_function(x[0], x[1])  # Apply ranking function
    )

    # ✅ Return the best match dynamically
    return [sorted_results[0][0]]  # Select best-ranked API dynamically

def get_relevant_api_endpoints(query, n_results=3):
    """Finds the closest API endpoints for a given query using ChromaDB."""
    search_results = collection.query(query_texts=[query], n_results=n_results)

    if not search_results or "metadatas" not in search_results or not search_results["metadatas"]:
        print("⚠️ No matching API endpoints found in ChromaDB.")
        return []

    return search_results["metadatas"][0]  # Return metadata of best matches

def generate_openai_function_call(intent, parameters, chroma_collection, llm_client, extracted_entities):
    """
    Dynamically selects an API endpoint based on the detected intent.
    """
    matched_clusters = match_query_to_cluster(intent, chroma_collection, extracted_entities)

    if not matched_clusters:
        raise ValueError(f"❌ No relevant API endpoint found for intent: {intent}")

    function_schemas = []
    for i, cluster_data in enumerate(matched_clusters):
        if not isinstance(cluster_data, dict):
            print(f"⚠️ Skipping invalid cluster {i}: {repr(cluster_data)}")
            continue

        description = cluster_data.get("description", "Unknown API Call")
        endpoint = cluster_data.get("endpoint", None)
        method = cluster_data.get("method", "GET")
        param_raw = cluster_data.get("parameters", "[]")

        if not endpoint:
            print(f"⚠️ Skipping cluster {i} - missing endpoint: {cluster_data}")
            continue

        # ✅ Ensure parameters are parsed correctly
        if isinstance(param_raw, str):
            try:
                param_list = json.loads(param_raw)
            except json.JSONDecodeError:
                param_list = []
        elif isinstance(param_raw, list):
            param_list = param_raw
        else:
            param_list = []

        # ✅ Inject extracted entities dynamically into API parameters
        final_parameters = {}
        for param in param_list:
            param_name = param["name"]
            final_parameters[param_name] = extracted_entities.get(param_name, f"from_user_input:{param_name}")

        # ✅ Append API Call Info
        function_schemas.append({
            "type": "function",
            "function": {
                "name": f"select_api_function_{i}",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param["name"]: {"type": param.get("schema", {}).get("type", "string")}
                        for param in param_list
                    },
                    "required": [param["name"] for param in param_list if param.get("required", False)]
                }
            },
            "endpoint": endpoint,
            "method": method,
            "parameters": final_parameters  # ✅ Inject correct parameters
        })

    if function_schemas:
        return function_schemas[0]
    else:
        print("⚠️ No valid API schemas generated.")
        return None

# Sequential execution using planner and function calls
def execute_planned_steps(planner_agent, query, extracted_entities):
    """
    Executes API calls based on dynamically generated plans.
    """
    execution_plan = planner_agent.generate_plan(query, extracted_entities)

    if not execution_plan:
        print("⚠️ No valid execution plan found.")
        return {}

    for step in execution_plan["plan"]:
        api_call_info = generate_openai_function_call(
            intent=step["intent"],
            parameters=step["parameters"],
            chroma_collection=planner_agent.chroma_collection,
            llm_client=planner_agent.llm_client,  # ✅ Removed incorrect intent_analyzer
            extracted_entities=extracted_entities  # ✅ Pass extracted entities correctly
        )

        if not api_call_info:
            print("❌ ERROR: No valid API call found.")
            return {}

        response = execute_api_call(api_call_info, extracted_entities)  # ✅ Pass extracted_entities

        return response

def dynamic_refinement_step(parameters):
    """
    Dynamically filters a list of items based on provided filter criteria.
    This avoids hardcoding refinement logic.
    """
    input_list_key = next((key for key, value in parameters.items() if isinstance(value, list)), None)
    if not input_list_key:
        print("⚠️ No valid list found for refinement.")
        return {}

    input_list = parameters[input_list_key]
    filter_criteria = {key: value for key, value in parameters.items() if key != input_list_key}

    # Apply filters dynamically based on the query parameters
    filtered_items = [
        item for item in input_list
        if all(
            (isinstance(value, list) and item.get(key) in value) or
            (item.get(key) == value)
            for key, value in filter_criteria.items()
        )
    ]

    # Remove duplicates if they have an 'id' field
    unique_items = {item["id"]: item for item in filtered_items}.values() if "id" in filtered_items[0] else filtered_items

    return {input_list_key: list(unique_items)}

def map_and_validate_api_call(step, shared_state, chroma_collection, llm_client):
    intent = step["intent"]
    parameters = step["parameters"].copy()

    # Resolve dependencies dynamically
    if step.get("depends_on"):
        dependency_step_num = step["depends_on"]
        dependency_response = shared_state.get(dependency_step_num)

        if not dependency_response:
            raise ValueError(f"Dependency step {dependency_step_num} unresolved for step {step['step']}")

        # Resolve parameters dynamically using LLM
        for param, value in parameters.items():
            if isinstance(value, str) and value.startswith("from_step_"):
                resolved_value = resolve_dependency_via_llm(
                    previous_response=dependency_response,
                    intent=intent,
                    param_name=param,
                    llm_client=llm_client
                )
                parameters[param] = resolved_value

    api_call_data = generate_openai_function_call(
        intent=intent,
        parameters=parameters,
        chroma_collection=chroma_collection,
        llm_client=llm_client
    )

    required_keys = ["endpoint", "method"]
    if not all(key in api_call_data for key in required_keys):
        raise ValueError(f"Invalid API call info from mapping: {api_call_data}")

    return api_call_data

def resolve_dependency(previous_response, json_path):
    keys = json_path.split('.')
    value = previous_response
    try:
        for key in keys:
            if key.isdigit():
                key = int(key)
            value = value[key]
        return value
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Error resolving dependency at '{json_path}': {e}")

def prepare_endpoint(endpoint, parameters):
    """
    Replaces placeholders in API endpoints with extracted entity values dynamically.
    """
    formatted_endpoint = endpoint
    for param, value in parameters.items():
        if f"{{{param}}}" in formatted_endpoint:
            formatted_endpoint = formatted_endpoint.replace(f"{{{param}}}", str(value))
    
    return formatted_endpoint

def execute_api_call(api_call_info, extracted_entities):
    """
    Executes an API call with dynamically injected parameters from extracted entities.
    """

    endpoint = api_call_info["endpoint"]
    method = api_call_info["method"].upper()
    params = api_call_info.get("parameters", {})

    # ✅ Inject extracted entity values dynamically
    for param_name in params.keys():
        if param_name in extracted_entities:
            params[param_name] = extracted_entities[param_name]

    # ✅ Handle Path vs Query Parameters
    formatted_endpoint = prepare_endpoint(endpoint, params)  # ✅ Inject path params
    query_params = {k: v for k, v in params.items() if "from_user_input" not in str(v)}

    url = f"https://api.themoviedb.org/3{formatted_endpoint}"
    headers = {"Authorization": f"Bearer {TMDB_API_KEY}"}

    print(f"🔍 Making API Call: {url} with params: {json.dumps(query_params, indent=2)}")

    response = requests.request(method, url, headers=headers, params=query_params)
    response.raise_for_status()
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

def detect_intents(query, chroma_collection, intent_analyzer):
    """
    Dynamically selects the most relevant API from ChromaDB using embeddings and extracted entities.
    """
    extracted_entities = intent_analyzer.extract_entities(query)  # ✅ Extract entities first

    matched_apis = match_query_to_cluster(query, chroma_collection, extracted_entities)  # ✅ Pass entities

    if not matched_apis:
        print("⚠️ No matching APIs found for:", query)
        return {}

    best_match = matched_apis[0]

    # ✅ Ensure parameters are structured correctly and pass `extracted_entities`
    parameters = extract_required_parameters(best_match["parameters"], extracted_entities)  # ✅ Fix applied

    return {
        "intent": best_match["description"],
        "endpoint": best_match["endpoint"],
        "method": best_match["method"],
        "parameters": parameters  # ✅ Inject extracted entities dynamically into API call
    }

def refine_embedding_input(query, extracted_entities):
    """
    Improves query embedding by injecting extracted entity types for better ChromaDB separation.
    """
    entity_labels = " ".join([f"{key}:{value}" for key, value in extracted_entities.items()])
    return f"{query} | Entities: {entity_labels}"  # ✅ Append entity metadata to avoid misclassification

def extract_required_parameters(api_parameters, extracted_entities):
    """
    Extracts required parameters dynamically from API metadata and injects extracted entities.
    """
    extracted_params = {}

    if isinstance(api_parameters, str):
        try:
            api_parameters = json.loads(api_parameters)
        except json.JSONDecodeError:
            print("❌ Failed to decode API parameters:", api_parameters)
            return {}

    if isinstance(api_parameters, list):
        for param in api_parameters:
            param_name = param.get("name")
            if param.get("required", False):
                # ✅ Auto-fill extracted entities if available
                extracted_params[param_name] = extracted_entities.get(param_name, f"from_user_input:{param_name}")

    return extracted_params

def main():
    llm_client = OpenAILLMClient(api_key=OPENAI_API_KEY)
    intent_analyzer = IntentAnalyzer(llm_client=llm_client)  # ✅ Initialize IntentAnalyzer
    planner_agent = PlannerAgent(llm_client=llm_client, chroma_collection=collection, intent_analyzer=intent_analyzer)  # ✅ Pass intent_analyzer

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        extracted_entities = intent_analyzer.extract_entities(user_query)  # ✅ Extract entities
        print(f"🧐 Extracted Entities: {extracted_entities}")
        intent_data = detect_intents(user_query, collection, intent_analyzer)  # ✅ Ensure intent_analyzer is passed

        if not intent_data:
            print("❌ No valid API found for the query.")
            continue

        execution_plan = planner_agent.generate_plan(user_query, extracted_entities)  # ✅ Generate execution plan

        if not execution_plan or not execution_plan.get("plan"):
            print("❌ No valid execution plan generated.")
            continue
        
        shared_state = execute_planned_steps(planner_agent, user_query, extracted_entities)  

        print(json.dumps(shared_state, indent=2))

if __name__ == "__main__":
    main()


