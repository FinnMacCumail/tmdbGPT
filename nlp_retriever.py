import json
import chromadb
import requests
import os
import spacy
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load API keys
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./semantic_chroma_db")
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

# Load NLP and embedding models
nlp = spacy.load("en_core_web_trf")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class IntentAnalyzer:
    """Extracts entities and determines API intents dynamically."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.nlp = spacy.load("en_core_web_trf")

    def extract_entities(self, query):
        """Extracts named entities dynamically from a query."""
        doc = self.nlp(query)
        entity_mappings = {
            "PERSON": "person",
            "WORK_OF_ART": "query",
            "DATE": "date",
            "GPE": "region"
        }
        extracted = {entity_mappings.get(ent.label_, ent.label_): ent.text for ent in doc.ents}
        print(f"🧐 Extracted Entities: {json.dumps(extracted, indent=2)}")
        return extracted


class OpenAILLMClient:
    """Uses OpenAI LLM to generate execution plans dynamically."""

    def __init__(self, api_key, model="gpt-4-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_response(self, prompt):
        """Generates a response using OpenAI with logging."""
        print(f"📝 LLM Prompt:\n{prompt}\n")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}]
        )
        output = response.choices[0].message.content.strip()
        print(f"🤖 LLM Response:\n{output}\n")
        return output


class PlannerAgent:
    def __init__(self, llm_client, chroma_collection, intent_analyzer):
        self.llm_client = llm_client
        self.chroma_collection = chroma_collection
        self.intent_analyzer = intent_analyzer

    def generate_plan(self, query, extracted_entities):
        """Dynamically generates an execution plan based on extracted entities and API intent detection."""
        matched_apis = self.match_query_to_cluster(query, extracted_entities)

        if not matched_apis:
            print("⚠️ No matching APIs found for:", query)
            return {"plan": []}

        steps = []
        for idx, api_data in enumerate(matched_apis, start=1):
            step = {
                "step": idx,
                "type": "api_call",
                "intent": api_data.get("description"),
                "endpoint": api_data.get("path"),
                "method": api_data.get("method"),
                "parameters": extract_required_parameters(api_data.get("query_params", []), extracted_entities),
                "depends_on": None
            }
            steps.append(step)
        print(f"✅ Execution Plan:\n{json.dumps(steps, indent=2)}\n")
        return {"plan": steps}

    def match_query_to_cluster(self, query, extracted_entities, n_results=5):
        """Selects the best-matching API dynamically based on embeddings and entity context."""
        refined_query = refine_embedding_input(query, extracted_entities)
        search_results = self.chroma_collection.query(query_texts=[refined_query], n_results=n_results)

        if not search_results or "metadatas" not in search_results or not search_results["metadatas"]:
            print("⚠️ No metadata found in ChromaDB results.")
            return []

        metadata_results = search_results["metadatas"][0]
        distances = search_results["distances"][0]

        sorted_results = sorted(
            zip(metadata_results, distances),
            key=lambda x: rank_function(x[0], x[1], extracted_entities)
        )

        return [sorted_results[0][0]] if sorted_results else []



    def generate_openai_function_call(self, step, extracted_entities):
        """
        Dynamically selects an API endpoint based on the step intent and extracted entities.
        This function ensures that the correct API path, method, and parameters are used.
        """
        matched_clusters = self.match_query_to_cluster(step["intent"], extracted_entities)

        if not matched_clusters:
            return None

        best_match = matched_clusters[0]

        query_params = best_match.get("query_params", "")
        if isinstance(query_params, str):
            query_params = [{"name": param.strip()} for param in query_params.split(", ") if param]  # Convert to list of dicts

        return {
            "endpoint": best_match["path"],
            "method": best_match["method"],            
            "parameters": extract_required_parameters(query_params, extracted_entities)
        }

def refine_embedding_input(query, extracted_entities):
    """Enhances query embedding by appending extracted entities to the query text."""
    entity_metadata = " | ".join([f"{key}:{value}" for key, value in extracted_entities.items()])
    return f"{query} | Entities: {entity_metadata}"


def extract_required_parameters(api_parameters, extracted_entities):
    """
    Dynamically injects extracted entities into API parameters, handling missing keys safely.
    """
    extracted_params = {}

    # ✅ Ensure `api_parameters` is a list of dictionaries
    if isinstance(api_parameters, str):
        api_parameters = [{"name": param.strip()} for param in api_parameters.split(", ") if param]

    if isinstance(api_parameters, list):
        for param in api_parameters:
            param_name = param.get("name")

            if param_name:  # ✅ Avoid processing None values
                extracted_params[param_name] = extracted_entities.get(param_name, None)

            # Ensure correct data type (fallback to sensible defaults)
            param_type = param.get("schema", {}).get("type", "string") if isinstance(param, dict) else "string"
            param_value = extracted_params.get(param_name, None)  

            if param_type == "integer":
                try:
                    extracted_params[param_name] = int(param_value) if param_value is not None else 1
                except ValueError:
                    extracted_params[param_name] = 1

    print(f"🛠️ Extracted Parameters (Updated): {json.dumps(extracted_params, indent=2)}")
    return extracted_params

def execute_api_call(api_call_info, extracted_entities):
    """Executes an API call with dynamically injected parameters."""
    url = f"https://api.themoviedb.org/3{api_call_info['endpoint']}"
    headers = {"Authorization": f"Bearer {TMDB_API_KEY}"}
    params = api_call_info.get("parameters", {})

    # ✅ Remove unresolved placeholders
    for param in list(params.keys()):
        if params[param] is None:
            del params[param]

    print(f"🔍 Making API Call: {url} with params: {json.dumps(params, indent=2)}")
    
    response = requests.request(api_call_info["method"].upper(), url, headers=headers, params=params)
    return response.json()


def execute_planned_steps(planner_agent, query, extracted_entities):
    """
    Executes API calls dynamically while resolving dependencies.
    - If an API requires an ID, fetch it first via a query-based API.
    - Runs steps sequentially, updating shared_state dynamically.
    """

    execution_plan = planner_agent.generate_plan(query, extracted_entities)

    if not execution_plan or not execution_plan.get("plan"):
        print("⚠️ No valid execution plan found.")
        return {}

    shared_state = {}

    for step in execution_plan["plan"]:
        print(f"\n🚀 Executing Step {step['step']} - {step['endpoint']}")

        step_query = step.get("intent", query)
        step_entities = planner_agent.intent_analyzer.extract_entities(step_query)
        combined_entities = {**extracted_entities, **step_entities}

        # ✅ If step requires ID, resolve it first
        placeholders = re.findall(r"{(.*?)}", step["endpoint"])
        missing_ids = [p for p in placeholders if f"{p}" not in combined_entities]

        if missing_ids:
            for missing_id in missing_ids:
                print(f"⚠️ Missing {missing_id}, searching for it first...")

                # ✅ Find the best alternative query-based API
                fallback_query_api = next((api for api in planner_agent.match_query_to_cluster(query, extracted_entities)
                                           if api.get("query_params") and "{" not in api["path"]), None)

                if fallback_query_api:
                    search_params = extract_required_parameters(fallback_query_api.get("query_params", []), combined_entities)
                    search_response = execute_api_call({
                        "endpoint": fallback_query_api["path"],
                        "method": fallback_query_api["method"],
                        "parameters": search_params
                    }, combined_entities)

                    # ✅ Extract the ID from query-based response
                    if "results" in search_response and search_response["results"]:
                        resolved_id = search_response["results"][0].get("id")
                        if resolved_id:
                            combined_entities[missing_id] = resolved_id
                            print(f"✅ Resolved {missing_id}: {resolved_id}")

        # ✅ Generate API Call
        api_call_info = planner_agent.generate_openai_function_call(step, combined_entities)

        if not api_call_info:
            print(f"❌ No valid API call found for step {step['step']}.")
            continue

        # ✅ Execute API Call
        response = execute_api_call(api_call_info, combined_entities)
        shared_state[step["step"]] = response  # Store response for future steps

        print(f"✅ Step {step['step']} Response Stored: {json.dumps(response, indent=2)}")

    return shared_state

def rank_function(item, score, extracted_entities):
    """
    Dynamically ranks API endpoints based on entity presence and query type.
    """
    endpoint = item.get("path", "")
    query_params = item.get("query_params", "")
    placeholders = re.findall(r"{(.*?)}", endpoint)
    weight = 0

    # ✅ Prioritize entity-aligned APIs
    for entity in extracted_entities:
        if entity in endpoint:
            weight -= 2  # Boost APIs that match entity type

    # ✅ If API is query-based (e.g., /search/person) and we have query params, prioritize it
    if query_params and not placeholders:
        weight -= 3  # Prioritize query-driven APIs when no ID is available

    # ✅ Penalize APIs requiring an ID if the ID is missing
    if placeholders and not any(f"{p}" in extracted_entities for p in placeholders):
        weight += 3  # Lower priority if ID is missing

    return score + weight  # Lower score is better


def main():
    llm_client = OpenAILLMClient(api_key=OPENAI_API_KEY)
    intent_analyzer = IntentAnalyzer(llm_client=llm_client)
    planner_agent = PlannerAgent(llm_client=llm_client, chroma_collection=collection, intent_analyzer=intent_analyzer)

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        extracted_entities = intent_analyzer.extract_entities(user_query)
        shared_state = execute_planned_steps(planner_agent, user_query, extracted_entities)
        print(json.dumps(shared_state, indent=2))


if __name__ == "__main__":
    main()
