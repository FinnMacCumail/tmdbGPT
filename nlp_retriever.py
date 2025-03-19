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
import traceback


# Load API keys
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

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
        """Parameter resolution with detailed tracing"""
        print(f"\n=== PARAMETER RESOLUTION ===")
        print(f"🔧 Context: {context}")
        print(f"📦 Available Entities: {json.dumps(entities, indent=2)}")
        print(f"🔧 Parameters to resolve: {json.dumps(params, indent=2)}")
        
        resolved = {}
        for param in params:
            if not isinstance(param, dict):
                print(f"⚠️ Skipping invalid parameter: {param}")
                continue

            pname = param.get("name")
            p_in = param.get("in")
            p_type = param.get("schema", {}).get("type", "string")

            print(f"\n🔍 Processing parameter: {pname} (in: {p_in}, type: {p_type})")

            # Query parameter resolution
            if p_in == "query":
                self._resolve_query_param(pname, entities, resolved)
            
            # Path parameter resolution
            elif p_in == "path":
                self._resolve_path_param(pname, entities, resolved)
            
            # Handle other parameter types
            else:
                print(f"⚠️ Unhandled parameter location: {p_in}")

        print(f"\n✅ Resolved parameters: {json.dumps(resolved, indent=2)}")
        return resolved
    
    def _resolve_query_param(self, name: str, entities: dict, resolved: dict):
        """Resolve query parameters with pattern matching"""
        if name == "query":
            print("🔎 Looking for query sources...")
            for key in entities:
                if key.endswith("_query"):
                    value = requests.utils.quote(str(entities[key]))
                    resolved[name] = value
                    print(f"✅ Found query source: {key} → {value}")
                    return
            print("⚠️ No matching query sources found")
        else:
            print(f"🔎 Looking for direct match: {name}")
            if name in entities:
                resolved[name] = entities[name]
                print(f"✅ Direct match found: {entities[name]}")
            else:
                print(f"⚠️ No direct match found")

    def _resolve_path_param(self, name: str, entities: dict, resolved: dict):
        """Resolve path parameters using naming convention"""
        print(f"🔎 Resolving path parameter: {name}")
        if "_id" in name:
            base_type = name.split("_")[0]
            id_key = f"{base_type}_id"
            print(f"🔍 Looking for ID: {id_key}")
            
            if id_key in entities:
                resolved[name] = entities[id_key]
                print(f"✅ ID found: {entities[id_key]}")
            else:
                print(f"⚠️ Missing ID: {id_key}")
        else:
            print(f"⚠️ Unrecognized path parameter pattern: {name}")

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
        """Debug-enhanced plan generation"""
        print("\n=== PLANNER INIT ===")
        print(f"📥 Received entities type: {type(entities)}")
        print(f"📥 Entity keys: {list(entities.keys())}")
        
        try:
            print("\n🔍 Starting vector search...")
            raw_steps = self._match_apis(query, entities)
            print(f"🔍 Found {len(raw_steps)} raw steps from vector search")
            
            final_plan = []
            for idx, step in enumerate(raw_steps, 1):
                print(f"\n🔧 Processing step {idx}: {json.dumps(step, indent=2)}")
                
                if not isinstance(step, dict):
                    print(f"⚠️ Invalid step type: {type(step)}")
                    continue
                    
                path = step.get("path")
                method = step.get("method")
                
                print(f"🔧 Path: {path}, Method: {method}")
                
                if not path or not method:
                    print("⚠️ Skipping invalid step (missing path/method)")
                    continue
                
                final_plan.append({
                    "endpoint": path,
                    "method": method,
                    "parameters": {},
                    "requires_resolution": "{" in path
                })
            
            print("\n✅ Valid steps in plan:")
            for step in final_plan:
                print(f"  - {step['endpoint']}")
            
            return {"plan": final_plan[:3]}
            
        except Exception as e:
            print(f"🚨 Critical planning error: {str(e)}")
            print(traceback.format_exc())
            return {"plan": []}
    
    def _resolve_parameters(self, params: list, entities: dict) -> dict:
        """Resolve parameters with logging"""
        resolved = {}
        for param in params:
            pname = param.get("name")
            p_in = param.get("in")
            
            print(f"🐞 Processing param: {pname} (in: {p_in})")
            
            if p_in == "query":
                # Find matching query source
                for key in entities:
                    if key.endswith("_query") and pname == "query":
                        value = requests.utils.quote(entities[key])
                        resolved[pname] = value
                        print(f"✅ Resolved query param from {key} → {value}")
                        break
            
            elif p_in == "path":
                # Match path parameters using naming convention
                base_type = pname.split("_")[0]
                id_key = f"{base_type}_id"
                if id_key in entities:
                    resolved[pname] = entities[id_key]
                    print(f"✅ Resolved path param from {id_key} → {entities[id_key]}")
        
        print(f"🐞 Final parameters: {json.dumps(resolved, indent=2)}")
        return resolved
    
    def _prioritize_steps(self, steps: List[Dict], entities: Dict) -> List[Dict]:
        """Prioritize steps with context awareness"""
        prioritized = []
        
        # Stage 1: Search endpoints
        print("\n🐞 [PLANNER] Prioritizing search endpoints...")
        search_steps = [s for s in steps if "/search/" in s["path"]]
        for step in search_steps:
            if any(ent in step["path"] for ent in entities):
                print(f"  ➕ Adding search endpoint: {step['path']}")
                prioritized.append(step)
        
        # Stage 2: Entity detail endpoints
        print("\n🐞 [PLANNER] Looking for detail endpoints...")
        detail_steps = []
        for step in steps:
            path_params = re.findall(r"{(\w+)}", step["path"])
            if any(f"{p}_id" in entities for p in path_params):
                print(f"  ➕ Adding detail endpoint: {step['path']}")
                detail_steps.append(step)
        prioritized.extend(detail_steps)
        
        # Stage 3: Filter out irrelevant steps
        print("\n🐞 [PLANNER] Filtering irrelevant endpoints...")
        filtered = []
        required_params = ["person", "movie", "tv"]  # Entity types from schema
        for step in prioritized:
            if any(p in step["path"] for p in required_params):
                filtered.append(step)
            else:
                print(f"  ➖ Filtering out: {step['path']}")
        
        return filtered[:5]  # Return top 5 relevant steps
    
    def _build_dynamic_plan(self, steps: List[Dict], entities: Dict) -> List[Dict]:
        plan = []
        resolved_ids = set()
        
        for step in sorted(steps, key=lambda x: x["path"].count("{")):
            path_params = re.findall(r"{(\w+)}", step["path"])
            
            # Check if all path parameters are resolved
            if all(f"{p}_id" in entities for p in path_params):
                plan.append({
                    "endpoint": step["path"],
                    "method": step["method"],
                    "parameters": self._map_parameters(step["parameters"], entities),
                    "requires_resolution": bool(path_params)
                })
                resolved_ids.update(path_params)
                
            # Add search steps for unresolved entities
            elif not path_params and "query" in [p["name"] for p in step["parameters"]]:
                plan.insert(0, {
                    "endpoint": step["path"],
                    "method": step["method"],
                    "parameters": {"query": next(iter(entities.values()))[0]},
                    "requires_resolution": False
                })
                
        return {"plan": plan[:5]}  # Limit to 5 steps

    def _map_parameters(self, params: list, entities: dict) -> dict:
        return {
            p["name"]: entities[f"{p['name']}_id"] 
            if f"{p['name']}_id" in entities 
            else entities.get(p["name"])
            for p in params
        }

    def _create_priority_plan(self, steps: List[Dict], entities: Dict) -> List[Dict]:
        """Create validated execution steps"""
        valid_steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "path" not in step or "method" not in step:
                continue
                
            # Enforce step structure
            validated = {
                "endpoint": step["path"],
                "method": step.get("method", "GET"),
                "parameters": step.get("parameters", {}),
                "requires_resolution": "{" in step["path"]
            }
            valid_steps.append(validated)
        
        # Prioritization logic
        return sorted(
            valid_steps,
            key=lambda x: (
                -int("/search/" in x["endpoint"]),
                -int(x["requires_resolution"]),
                len(x["endpoint"])
            )
        )[:5]  # Return top 5 most relevant
    
    # In nlp_retriever.py's _match_apis method:
    def _match_apis(self, query: str, entities: Dict) -> List[Dict]:
        """Improved API matching with vector search insights"""
        print("\n=== VECTOR SEARCH DEBUG ===")
        print(f"🐞 Query: {query}")
        print(f"🐞 Entities: {json.dumps(entities, indent=2)}")
        
        # Get raw vector search results
        results = self.collection.query(
            query_texts=[query],
            n_results=10,
            include=["metadatas", "distances", "documents"]
        )
        
        # print("🔍 Raw vector search results:")
        # for i, (meta, dist, doc) in enumerate(zip(results["metadatas"][0], 
        #                                         results["distances"][0],
        #                                         results["documents"][0])):
        #     print(f"{i+1}. {meta['path']} (distance: {dist:.2f})")
        #     print(f"   Metadata: {json.dumps(meta, indent=4)}")
        #     print(f"   Document: {doc[:100]}...\n")

        # # Filter valid APIs
        # valid_apis = []
        # for meta in results["metadatas"][0]:
        #     if not isinstance(meta, dict): continue
        #     if self._is_valid_api(meta):
        #         valid_apis.append(meta)
        #         print(f"✅ Valid API: {meta['path']}")
        #     else:
        #         print(f"🚫 Invalid API: {meta.get('path', 'unknown')}")

        # return valid_apis
        return [
            meta for meta in results["metadatas"][0] 
            if isinstance(meta, dict) 
            and "path" in meta 
            and "method" in meta
        ]
    
    def _is_valid_api(self, api_meta: Dict) -> bool:
        """Validate API structure with logging"""
        valid = all(key in api_meta for key in ["path", "method"]) 
        valid &= isinstance(api_meta.get("parameters", []), list)
        
        if not valid:
            print(f"⚠️ Invalid API structure: {json.dumps(api_meta, indent=2)}")
        
        return valid

    def _parse_semantic_parameters(self, meta: Dict) -> List:
        """Convert semantic_embed parameters to original format"""
        param_map = {
            "search_fields": {"name": "query", "in": "query"},
            "temporal_filters": {"name": ["date.gte", "date.lte"], "in": "query"},
            # Add other parameter mappings from semantic_embed's analyze_parameters
        }
        
        return [param_map.get(k, {}) for k in meta.get("parameters", {})]

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

    def _has_required_ids(self, step: Dict, entities: Dict) -> bool:
        """Check path parameters against resolved entities"""
        path_params = re.findall(r"{(\w+)}", step["path"])
        return all(
            f"{param}_id" in entities or param in entities
            for param in path_params
        )

    def _add_resolution_steps(self, steps: List[Dict], entities: Dict) -> List[Dict]:
        """Safe resolution step addition with error handling"""
        resolved_plan = []
        for step in steps:
            # Validate step structure first
            if not self._is_valid_step(step):
                continue

            # Existing resolution logic...
            resolved_plan.append(step)
        return resolved_plan

    def _is_valid_step(self, step: Dict) -> bool:
        """Validate step structure"""
        required_keys = {"path", "method", "parameters"}
        return all(key in step for key in required_keys)
    
    def _create_search_step(self, entity_type: str, entities: dict) -> Optional[Dict]:
        """Create search step using query terms"""
        return {
            "step": f"search_{entity_type}",
            "type": "api_call",
            "endpoint": f"/search/{entity_type}",
            "method": "GET",
            "parameters": {"query": entities[f"{entity_type}_query"]}
        }
    
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
    try:
        # Dynamic path parameter substitution
        endpoint = api_call_info["endpoint"]
        for match in re.finditer(r"{(\w+)}", endpoint):
            param = match.group(1)
            endpoint = endpoint.replace(match.group(0), str(entities.get(f"{param}_id", "")))
        
        response = requests.request(
            api_call_info["method"],
            f"{BASE_URL}{endpoint}",
            headers=HEADERS,
            params=api_call_info.get("parameters", {})
        )
        
        return response.json() if response.status_code == 200 else {
            "error": f"API Error {response.status_code}",
            "details": response.text
        }
    except Exception as e:
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
