import json
import chromadb
import requests
import os
import spacy
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import re
from datetime import datetime
from typing import Dict, List, Optional


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

# TMDB API configuration
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TMDB_API_KEY}"}


class EnhancedIntentAnalyzer:
    """Enhanced entity recognition with genre and temporal analysis"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self._add_custom_patterns()
        self.genre_map = self._fetch_genre_mappings()
    
    def _add_custom_patterns(self):
        """Add patterns for genre and date recognition"""
        ruler = self.nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "GENRE", "pattern": [{"LOWER": {"IN": ["action", "comedy", "drama"]}}]},
            {"label": "DATE", "pattern": [{"SHAPE": "dddd"}]},  # Years like 2023
            {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{4}s"}}]}  # Decades like 2010s
        ]
        ruler.add_patterns(patterns)
    
    def _fetch_genre_mappings(self) -> Dict[str, int]:
        """Fetch genre IDs from TMDB"""
        response = requests.get(f"{BASE_URL}/genre/movie/list", headers=HEADERS)
        return {genre["name"].lower(): genre["id"] for genre in response.json().get("genres", [])}
    
    def extract_entities(self, query: str) -> Dict[str, str]:
        """Extract entities with enhanced recognition"""
        doc = self.nlp(query)
        entities = {
            "person": [], "title": [], "genre": [], "date": [],
            "year": [], "region": [], "numeric": []
        }
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["person"].append(ent.text)
            elif ent.label_ == "WORK_OF_ART":
                entities["title"].append(ent.text)
            elif ent.label_ == "GPE":
                entities["region"].append(ent.text)
            elif ent.label_ == "DATE":
                if ent.text.endswith("s"):
                    entities["date"].append(ent.text[:-1])  # Remove 's' from decades
                else:
                    entities["date"].append(ent.text)
            elif ent.label_ == "GENRE":
                entities["genre"].append(ent.text)
            elif ent.label_ == "CARDINAL":
                entities["numeric"].append(ent.text)
        
        # Extract year from dates
        entities["year"] = [d for d in entities["date"] if d.isdigit() and len(d) == 4]
        
        print(f"🔍 Extracted Entities: {json.dumps(entities, indent=2)}")
        return entities

class ContextAwareParameterHandler:
    """Handles parameter mapping with endpoint context awareness"""
    
    def __init__(self, genre_map: Dict[str, int]):
        self.genre_map = genre_map
    
    def resolve_parameters(self, params: list, entities: dict, context: str) -> dict:
        """Dynamic parameter injection without path assumptions"""
        resolved = {}
        
        for param in params:
            pname = param.get("name", "")
            
            # Match entity type to parameter name
            for entity_type, values in entities.items():
                if entity_type in pname and values:
                    resolved[pname] = values[0]
                    break
                    
            # Special handling for generic "query" param
            if pname == "query" and not resolved.get(pname):
                resolved[pname] = next(iter(entities.values()), [""])[0]
                
        return resolved
    
    def _get_primary_entity(self, context: str) -> Optional[str]:
        """Extract primary entity type from endpoint context"""
        match = re.search(r"for (\w+) entities", context)
        return match.group(1).lower() if match else None
    
    def _resolve_query_value(self, entities: Dict, primary_entity: Optional[str]) -> Optional[str]:
        """Resolve query value with priority: primary entity > title > person"""
        if primary_entity and entities.get(primary_entity):
            return entities[primary_entity][0]
        if entities.get("title"):
            return entities["title"][0]
        if entities.get("person"):
            return entities["person"][0]
        return None
    
    def _format_date(self, date_str: str, param_name: str) -> str:
        """Convert date entities to TMDB-compatible formats"""
        if param_name.endswith(".gte"):
            return f"{date_str}-01-01"
        if param_name.endswith(".lte"):
            return f"{date_str}-12-31"
        return date_str

class IntelligentPlanner:
    """Generates execution plans with context-aware resolution"""
    
    def __init__(self, chroma_collection, intent_analyzer):
        self.collection = chroma_collection
        self.intent_analyzer = intent_analyzer
        self.param_handler = ContextAwareParameterHandler(intent_analyzer.genre_map)

    def generate_plan(self, query: str, entities: Dict) -> Dict:
        """Generate execution plan with parameter deserialization"""
        matched_apis = self._match_apis(query, entities)
        plan = []
        
        for idx, api_data in enumerate(matched_apis, 1):
            # Deserialize parameters if stored as JSON string
            parameters = api_data.get("parameters", [])
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    parameters = []
            
            step = {
                "step": idx,
                "type": "api_call",
                "endpoint": api_data["path"],
                "method": api_data["method"],
                "parameters": self.param_handler.resolve_parameters(
                    parameters,  # Now properly deserialized
                    entities,
                    api_data.get("context", "")
                ),
                "requires_resolution": api_data.get("requires_resolution", False)
            }
            plan.append(step)
        
        return {"plan": self._add_resolution_steps(plan, entities)}

    def _match_apis(self, query: str, entities: Dict) -> List[Dict]:
        """Robust API matching with metadata validation"""
        results = self.collection.query(query_texts=[query], n_results=5)
        
        valid_apis = []
        # Handle ChromaDB's nested response format
        for metadata in results.get("metadatas", [[]])[0]:  # First query text results
            if isinstance(metadata, dict):
                valid_apis.append(metadata)
            elif isinstance(metadata, str):
                try:
                    parsed = json.loads(metadata)
                    if isinstance(parsed, dict):
                        valid_apis.append(parsed)
                except json.JSONDecodeError:
                    continue
        
        # Validate required fields
        return [
            api for api in valid_apis
            if isinstance(api.get("path"), str) 
            and isinstance(api.get("method"), str)
            and "parameters" in api
        ]

    def _sort_results(self, metadatas: List[Dict], distances: List[float], entities: Dict):
        """Safe sorting with type checking"""
        return sorted(
            zip(metadatas, distances),
            key=lambda x: self._ranking_score(x[0], x[1], entities)
        )[:3]  # Return top 3 results

    def _ranking_score(self, api_data: dict, distance: float, entities: dict) -> float:
        score = float(distance)
        params = json.loads(api_data.get("parameters", "[]"))
        
        # Boost for query parameter match
        if any(p.get("name") == "query" for p in params):
            if "person" in entities and "query" in [p["name"] for p in params]:
                score -= 10.0  # Massive boost for person search endpoints
            else:
                score -= 5.0  # General search boost
                
        # Boost for entity-specific filters
        for entity_type in entities:
            if any(entity_type in p["name"] for p in params):
                score -= 3.0
                
        return score

    def _has_required_ids(self, api_data: Dict, entities: Dict) -> bool:
        """Check if required path parameters exist in entities"""
        path_params = api_data.get("path_params", [])
        return all(
            any(param in key for key in entities.keys())
            for param in path_params
        )

    def _add_resolution_steps(self, plan: List[Dict], entities: Dict) -> List[Dict]:
        """Add ID resolution steps where needed"""
        resolved_plan = []
        for step in plan:
            if step["requires_resolution"] and not self._has_required_ids(step, entities):
                resolution_step = self._create_resolution_step(step, entities)
                if resolution_step:
                    resolved_plan.append(resolution_step)
            resolved_plan.append(step)
        return resolved_plan

    def _create_resolution_step(self, step: Dict, entities: Dict) -> Optional[Dict]:
        """Create ID resolution step for API dependencies"""
        placeholder = re.findall(r"{(\w+)}", step["endpoint"])[0]
        search_endpoint = self._find_search_endpoint(placeholder)
        
        if search_endpoint:
            return {
                "step": f"{step['step']}-resolve",
                "type": "resolution",
                "endpoint": search_endpoint["path"],
                "method": search_endpoint["method"],
                "parameters": self.param_handler.resolve_parameters(
                    search_endpoint.get("parameters", []),
                    entities,
                    search_endpoint.get("context", "")
                )
            }
        return None

    def _find_search_endpoint(self, entity_type: str) -> Optional[Dict]:
        """Find search endpoint for a given entity type"""
        results = self.collection.query(
            query_texts=[f"Search endpoint for {entity_type}"],
            n_results=1
        )
        return results["metadatas"][0][0] if results["metadatas"] else None   
    
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


def extract_required_parameters(api_parameters, extracted_entities):
    """Robust parameter extraction with JSON deserialization and type safety"""
    extracted_params = {}
    
    # 1. Deserialize parameters if stored as JSON string
    if isinstance(api_parameters, str):
        try:
            api_parameters = json.loads(api_parameters)
        except json.JSONDecodeError:
            print("⚠️ Failed to deserialize parameters:", api_parameters)
            return {}

    # 2. Handle empty or invalid parameter formats
    if not isinstance(api_parameters, list):
        print("⚠️ Invalid parameters format, expected list")
        return {}

    # 3. Process each parameter with type validation
    for param in api_parameters:
        # Skip invalid parameter entries
        if not isinstance(param, dict):
            continue
            
        param_name = param.get("name", "").strip()
        if not param_name:
            continue

        # 4. Dynamic entity-parameter mapping
        param_found = False
        for entity_key, entity_value in extracted_entities.items():
            # Case-insensitive partial matching (e.g., "person" matches "person_id")
            if entity_key.lower() in param_name.lower():
                extracted_params[param_name] = entity_value
                param_found = True
                print(f"✅ Mapped {param_name} → {entity_value}")
                break

        # 5. Fallback for required parameters without direct matches
        if not param_found and param.get("required", False):
            print(f"⚠️ Missing required parameter: {param_name}")
            extracted_params[param_name] = None

    # 6. Clean null values and URL-encode strings
    final_params = {}
    for key, value in extracted_params.items():
        if value is not None:
            final_params[key] = requests.utils.quote(str(value)) if isinstance(value, str) else value

    print(f"🛠️ Final parameters: {json.dumps(final_params, indent=2)}")
    return final_params

def execute_api_call(api_call_info: Dict, entities: Dict) -> Dict:
    """Robust API execution with parameter validation"""
    try:
        # Validate metadata structure
        if not isinstance(api_call_info, dict):
            raise ValueError("Invalid API call info format")
            
        endpoint = api_call_info.get("endpoint", "")
        method = api_call_info.get("method", "GET")
        params = api_call_info.get("parameters", {})
        
        # Convert string parameters to dict if needed
        if isinstance(params, str):
            params = json.loads(params)
            
        # Encode parameters
        safe_params = {
            k: requests.utils.quote(v) if isinstance(v, str) else v
            for k, v in params.items()
        }
        
        response = requests.request(
            method,
            f"https://api.themoviedb.org/3{endpoint}",
            headers={"Authorization": f"Bearer {TMDB_API_KEY}"},
            params=safe_params
        )
        return response.json()
    except Exception as e:
        print(f"🚨 API Execution Failed: {str(e)}")
        return {"error": str(e)}

def execute_planned_steps(planner_agent, query, extracted_entities):
    """
    Executes API calls dynamically while resolving dependencies.
    - If an API requires an ID, fetch it first via a query-based API.
    - Runs steps sequentially, updating shared_state dynamically.
    """

    execution_plan = planner_agent.generate_plan(query, extracted_entities)
    if not isinstance(execution_plan, dict) or not isinstance(execution_plan.get("plan"), list):
        raise ValueError("Invalid execution plan format")
        
    shared_state = {}
    
    for step in execution_plan["plan"]:
        # Validate step structure
        if not isinstance(step, dict) or not all(key in step for key in ["endpoint", "method"]):
            print(f"⚠️ Skipping invalid step: {step}")
            continue
            
        print(f"\n🚀 Executing Step {step.get('step', '?')} - {step['endpoint']}")

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
        api_call_info = {
            "endpoint": step["endpoint"],
            "method": step["method"],
            "parameters": step["parameters"]
        }

        if not api_call_info:
            print(f"❌ No valid API call found for step {step['step']}.")
            continue

        # ✅ Execute API Call
        response = execute_api_call(api_call_info, combined_entities)
        shared_state[step["step"]] = response  # Store response for future steps

        print(f"✅ Step {step['step']} Response Stored: {json.dumps(response, indent=2)}")

    return shared_state

def main():
    intent_analyzer = EnhancedIntentAnalyzer()
    planner = IntelligentPlanner(collection, intent_analyzer)
    
    while True:
        try:
            query = input("Enter your query (or 'exit' to quit): ")
            if query.lower() in ["exit", "quit"]:
                break
                
            entities = intent_analyzer.extract_entities(query)
            # Pass the planner instance, not the plan
            results = execute_planned_steps(planner, query, entities)
            print(f"🎉 Final Results:\n{json.dumps(results, indent=2)}")
            
        except Exception as e:
            print(f"⚠️ Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
