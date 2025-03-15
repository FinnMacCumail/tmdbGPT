import json
import chromadb
import requests
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import spacy
from openai import OpenAI

# Load API keys
dotenv_path = os.path.join(os.getcwd(), ".env") 
load_dotenv(dotenv_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

nlp = spacy.load("en_core_web_trf")

class IntentAnalyzer:
    def __init__(self, llm_client=None):
        self.nlp = spacy.load("en_core_web_trf")
        self.llm_client = llm_client  

    def extract_entities(self, query):
        """Extracts named entities dynamically using NLP."""
        doc = nlp(query)
        extracted = {}
        entity_mappings = {
            "PERSON": "query",  
            "WORK_OF_ART": "query",  
            "DATE": "date",
            "GPE": "region"
        }
        for ent in doc.ents:
            if ent.label_ in entity_mappings:
                extracted[entity_mappings[ent.label_]] = ent.text
        return extracted
    
    def analyze(self, query):
        """
        Extracts entities and determines whether a multi-step plan is required.
        Uses LLM if needed.
        """
        entities = self.extract_entities(query)

        # Use LLM only if no clear intent is found
        prompt = (
            f"Decompose this query into a structured execution plan.\n"
            f"Query: \"{query}\"\n"
            f"Entities: {json.dumps(entities)}\n"
            "Return a structured JSON with a list of steps."
        )

        if self.llm_client:
            response = self.llm_client.generate_response(prompt)
            try:
                plan = json.loads(response)
                return {"entities": entities, "steps": plan.get("steps", [])}
            except Exception as e:
                print("LLM error:", e)

        return {"entities": entities, "intents": {}}

class OpenAILLMClient:
    def __init__(self, api_key, model="gpt-4-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, prompt):
        messages = [{"role": "system", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content.strip()

class PlannerAgent:
    def __init__(self, llm_client, chroma_collection, intent_analyzer):
        self.llm_client = llm_client
        self.chroma_collection = chroma_collection
        self.intent_analyzer = intent_analyzer

    def generate_plan(self, query):
        """
        Generates a multi-step execution plan for complex queries.
        """
        analysis = self.intent_analyzer.analyze(query)

        # If it's a multi-step query, return structured steps
        if "steps" in analysis:
            return {"plan": analysis["steps"]}

        # Otherwise, treat it as a single query
        return {"plan": [{
            "step": 1,
            "intent": "search_general",
            "endpoint": "/search/multi",
            "method": "GET",
            "parameters": analysis["entities"],
            "depends_on": None
        }]}

def execute_planned_steps(planner_agent, query):
    """
    Executes API calls based on the generated plan.
    Handles dependency resolution dynamically.
    """
    execution_plan = planner_agent.generate_plan(query)

    if not execution_plan.get("plan"):
        print("⚠️ No valid execution plan found.")
        return {}

    shared_state = {}

    for step in execution_plan["plan"]:
        # Resolve dependencies
        if step.get("depends_on") is not None:
            for param, value in step["parameters"].items():
                if isinstance(value, str) and value.startswith("from_step_"):
                    dependency_step = int(value.split("_")[-1])
                    step["parameters"][param] = shared_state.get(dependency_step, {}).get(param, "")

        # Execute API Call
        response = execute_api_call(step)
        shared_state[step["step"]] = response

    return shared_state

def execute_api_call(step):
    """
    Dynamically executes an API call using extracted parameters.
    """
    endpoint = step["endpoint"]
    method = step["method"].upper()
    params = step.get("parameters", {})

    # Construct API URL
    url = f"https://api.themoviedb.org/3{endpoint}"
    headers = {"Authorization": f"Bearer {TMDB_API_KEY}"}

    print(f"🔍 Making API Call: {url} with params: {json.dumps(params, indent=2)}")

    response = requests.request(method, url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def main():
    llm_client = OpenAILLMClient(api_key=OPENAI_API_KEY)
    intent_analyzer = IntentAnalyzer(llm_client=llm_client)
    planner_agent = PlannerAgent(llm_client, collection, intent_analyzer)

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        shared_state = execute_planned_steps(planner_agent, user_query)
        print(json.dumps(shared_state, indent=2))

if __name__ == "__main__":
    main()
