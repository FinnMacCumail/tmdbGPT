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
import requests
from requests.exceptions import RequestException
from utils.metadata_parser import MetadataParser
import networkx as nx
import time


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
        
        #print(f"🔍 Extracted Entities: {json.dumps(entities, indent=2)}")
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
    
    def __init__(self, chroma_collection, intent_analyzer, llm_client):
        self.collection = chroma_collection
        self.intent_analyzer = intent_analyzer
        self.llm_client = llm_client
        self.entity_registry = {}
        self.execution_graph = nx.DiGraph()

    def generate_plan(self, query: str, entities: Dict, intents: Dict) -> Dict:
        """Generate full plan structure with detailed debugging"""
        print(f"\n{'='*30} GENERATING FULL PLAN {'='*30}")
        print(f"📝 Initial Query: {query}")
        print(f"🔍 Detected Entities: {json.dumps(entities, indent=2)}")
        print(f"🎯 Detected Intents: {json.dumps(intents, indent=2)}")
        
        start_time = time.time()
        try:
            print("\n🔧 Starting plan validation process...")
            validated_steps = self._generate_validated_steps(query)
            plan = {
                "plan": validated_steps,
                "graph": self.execution_graph,
                "metadata": {
                    "generation_time": time.time() - start_time,
                    "step_count": len(validated_steps)
                }
            }
            
            print(f"\n✅ Successfully generated plan with {len(validated_steps)} steps")
            print(f"⏱️  Total generation time: {plan['metadata']['generation_time']:.2f}s")
            print(f"\n📋 Final Plan Structure:")
            print(json.dumps(plan, indent=2, default=str))
            
            return plan
            
        except Exception as e:
            print(f"\n{'⚠️'*10} PLAN GENERATION FAILED {'⚠️'*10}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            traceback.print_exc()
            return {
                "plan": [],
                "graph": nx.DiGraph(),
                "error": str(e)
            }
    
    def _generate_llm_plan(self, query: str) -> Dict:
        """Structured LLM planning with validation prompt"""
        prompt = f"""Generate a TMDB API execution plan for: "{query}"

Respond with JSON containing:
- Ordered steps with dependencies
- TMDB endpoint types (e.g., /search/person)
- Parameter definitions using $ placeholders
- Output entities to preserve

Available endpoints:
Search: /search/[movie|tv|person]
Details: /[movie|tv|person]/{{id}}
Relationships: /person/{{person_id}}/movie_credits
Discover: /discover/movie with filters

Example:
{{
  "plan": [
    {{
      "step_id": 1,
      "description": "Search for person",
      "endpoint_type": "/search/person",
      "parameters": {{"query": "Sofia Coppola"}},
      "output_entities": ["person_id"],
      "depends_on": []
    }}
  ]
}}"""

        response = self.llm_client.generate_response(prompt)
        return self._parse_llm_response(response)

    def _generate_validated_steps(self, query: str) -> List[Dict]:
        """Validation implementation with enhanced parameter checking and debugging"""
        print(f"\n{'='*30} VALIDATING STEPS {'='*30}")
        validated_steps = []
        
        try:
            # Generate initial LLM plan
            llm_plan = self._generate_llm_plan(query)
            print(f"\n📄 Raw LLM-generated Plan:")
            print(json.dumps(llm_plan, indent=2))
            
            if not llm_plan.get('plan'):
                print("⚠️ No steps generated by LLM")
                return []
                
            for idx, raw_step in enumerate(llm_plan.get('plan', [])):
                step_debug_header = f"\n🔍 STEP {idx+1} VALIDATION [Original: {raw_step.get('description')}]"
                print(f"{step_debug_header}")
                print("-"*len(step_debug_header))
                
                try:
                    # Validate endpoint existence
                    endpoint_match = self._find_endpoint_match(raw_step.get('endpoint_type', ''))
                    if not endpoint_match:
                        print(f"🚫 Step {idx+1} rejected - No matching endpoint found")
                        continue
                        
                    print(f"✅ Endpoint validated: {endpoint_match['path']}")
                    
                    # Get detailed parameter schema
                    supported_params = {p['name']: p for p in endpoint_match.get('parameters', [])}
                    required_params = [p['name'] for p in endpoint_match.get('parameters', []) 
                                    if p.get('required', False)]
                    
                    # Validate parameters with type checking
                    valid_params = {}
                    print(f"\n🔧 Parameter Validation Details:")
                    print(f"| {'Parameter':<20} | {'Status':<10} | {'Type':<15} | {'Value':<30} |")
                    print("|----------------------|------------|-----------------|--------------------------------|")
                    
                    for param, value in raw_step.get('parameters', {}).items():
                        param_meta = supported_params.get(param, {})
                        param_type = param_meta.get('schema', {}).get('type', 'unknown')
                        
                        if param in supported_params:
                            # Type validation
                            if param_type == 'integer' and not str(value).isdigit():
                                status = "TYPE_MISMATCH"
                            else:
                                valid_params[param] = value
                                status = "VALID"
                        else:
                            status = "UNSUPPORTED"
                        
                        print(f"| {param:<20} | {status:<10} | {param_type:<15} | {str(value):<30} |")
                    
                    # Check required parameters
                    missing_required = [p for p in required_params if p not in valid_params]
                    if missing_required:
                        print(f"\n🚨 Missing required parameters: {missing_required}")
                        raise ValueError(f"Missing required parameters: {missing_required}")
                    
                    # Build validated step
                    validated_step = {
                        "step_id": idx+1,
                        "description": raw_step.get('description', 'Unnamed step'),
                        "original_endpoint": raw_step.get('endpoint_type'),
                        "validated_endpoint": endpoint_match['path'],
                        "method": endpoint_match.get('method', 'GET'),
                        "parameters": valid_params,
                        "dependencies": raw_step.get('depends_on', []),
                        "output_entities": raw_step.get('output_entities', []),
                        "validation_status": "success",
                        "validation_timestamp": datetime.now().isoformat(),
                        "param_validation": {
                            "supported_params": list(supported_params.keys()),
                            "required_params": required_params,
                            "missing_required": missing_required
                        }
                    }
                    
                    if not valid_params:
                        print(f"🚫 No valid parameters - rejecting step")
                        raise ValueError("No valid parameters provided")
                    
                    validated_steps.append(validated_step)
                    print(f"\n✅ Step {idx+1} validation successful")
                    
                except Exception as step_error:
                    print(f"\n🚫 Step {idx+1} validation failed: {str(step_error)}")
                    validated_steps.append({
                        "step_id": idx+1,
                        "validation_status": "failed",
                        "error": str(step_error),
                        "raw_data": raw_step,
                        "validation_timestamp": datetime.now().isoformat()
                    })
                    
            print(f"\n📊 Validation Summary:")
            print(f"Total steps processed: {len(llm_plan.get('plan', []))}")
            print(f"Valid steps: {len([s for s in validated_steps if s['validation_status'] == 'success'])}")
            print(f"Failed steps: {len([s for s in validated_steps if s['validation_status'] == 'failed'])}")
            print(f"Required parameter failures: {sum(len(s.get('param_validation', {}).get('missing_required', [])) for s in validated_steps)}")
            
            return validated_steps
            
        except Exception as e:
            print(f"\n🚨 Critical validation error: {str(e)}")
            traceback.print_exc()
            return []

    def _parse_llm_response(self, response: str) -> Dict:
        """Safely parse LLM JSON output"""
        try:
            return json.loads(response.strip("`").replace("json\n", ""))
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse LLM response: {response}")
            return {"plan": []}

    def _validate_plan(self, plan: Dict) -> Dict:
        """Validate each step against ChromaDB embeddings"""
        validated_steps = []
        
        for step in plan.get("plan", []):
            print(f"\n🔍 Validating step {step['step_id']}: {step['description']}")
            
            # Find best matching endpoint in Chroma
            endpoint = self._find_endpoint_match(step['endpoint_type'])
            if not endpoint:
                print(f"⚠️ No Chroma match for {step['endpoint_type']}")
                continue
                
            # Map parameters to endpoint schema
            validated_params = self._validate_parameters(
                endpoint['parameters'], 
                step['parameters']
            )
            
            validated_steps.append({
                **step,
                "validated_endpoint": endpoint['path'],
                "validated_method": endpoint['method'],
                "resolved_parameters": validated_params
            })
            
        return {"plan": validated_steps}

    def _find_endpoint_match(self, endpoint_type: str) -> Optional[Dict]:
        """Find closest matching endpoint from ChromaDB"""
        results = self.collection.query(
            query_texts=[endpoint_type],
            n_results=1,
            include=["metadatas"]
        )
        return results['metadatas'][0][0] if results['metadatas'][0] else None

    def _validate_parameters(self, endpoint_meta: Dict, step_params: Dict) -> Dict:
        """Strict parameter validation with schema checking"""
        valid_params = {}
        supported_params = {p['name']: p for p in endpoint_meta.get('parameters', [])}
        
        print(f"🔍 Endpoint supports parameters: {list(supported_params.keys())}")
        
        for param, value in step_params.items():
            if param in supported_params:
                param_schema = supported_params[param].get('schema', {})
                
                # Type validation
                if param_schema.get('type') == 'integer' and not str(value).isdigit():
                    print(f"🚨 Type mismatch: {param} requires integer")
                    continue
                    
                valid_params[param] = value
                print(f"✅ Validated {param} ({param_schema.get('type', 'string')})")
            else:
                print(f"🚫 Rejected unsupported parameter: {param}")

        # Check required parameters
        required_params = [p['name'] for p in endpoint_meta.get('parameters', []) 
                        if p.get('required')]
        missing = [p for p in required_params if p not in valid_params]
        
        if missing:
            print(f"🚨 Missing required parameters: {missing}")
            
        return valid_params

    def _build_dependency_graph(self, plan: Dict):
        """Create execution graph from dependencies"""
        self.execution_graph.clear()
        
        for step in plan['plan']:
            step_id = step['step_id']
            self.execution_graph.add_node(step_id, **step)
            
            # Create edges for dependencies
            for dep_id in step.get('depends_on', []):
                if dep_id in self.execution_graph:
                    self.execution_graph.add_edge(dep_id, step_id)
                    
        print(f"\n🔗 Dependency Graph: {self.execution_graph.edges()}")
    
    def _needs_initial_search(self, entities: Dict, plan_steps: List[Dict]) -> bool:
        """Determines if search is required before execution"""
        # Check if any planned steps require unresolved IDs
        required_ids = set()
        for step in plan_steps:
            required_ids.update(re.findall(r"{(\w+_id)}", step["endpoint"]))
            
        # Check if we have gaps in entity resolution
        return not all(f"{id}" in entities for id in required_ids)
    
    def _find_search_steps(self, candidates: List[Dict], entities: Dict) -> List[Dict]:
        """Find search endpoints that resolve missing IDs"""
        missing_ids = self._get_missing_entity_ids(candidates, entities)
        return [
            step for step in candidates
            if self._is_search_endpoint(step) 
            and self._can_resolve_missing(step, missing_ids)
        ]
    
    def _get_missing_entity_ids(self, steps: List[Dict], entities: Dict) -> List[str]:
        """Identify unresolved IDs needed by steps"""
        required = set()
        for step in steps:
            required.update(re.findall(r"{(\w+_id)}", step["endpoint"]))
        return [id for id in required if f"{id}" not in entities]
    
    def _is_search_endpoint(self, step: Dict) -> bool:
        """Check if endpoint is search-oriented"""
        return (
            "/search/" in step["endpoint"] or 
            step["metadata"].get("operation_type") == "Entity Discovery"
        )
    
    def _can_resolve_missing(self, step: Dict, missing_ids: List[str]) -> bool:
        """Check if search endpoint can resolve needed IDs"""
        return any(
            id.replace("_id", "") in step["metadata"]["search_capable_for"]
            for id in missing_ids
        )

    def _find_executable_steps(self, candidates: List[Dict], entities: Dict) -> List[Dict]:
        """Find steps that can execute with current entities"""
        return [
            step for step in candidates
            if all(
                f"{param}" in entities
                for param in re.findall(r"{(\w+_id)}", step["endpoint"])
            )
        ]

    def _identify_dependencies(self, steps: List[Dict]) -> Dict:
        dep_map = {}
        for step in steps:
            key = f"{step['endpoint']}-{step['method']}"
            dep_map[key] = {
                "path_params": re.findall(r"{(\w+_id)}", step["endpoint"]),
                "query_params": [
                    p for p in step.get("parameters", {}) 
                    if p.endswith("_id")
                ]
            }
        return dep_map

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
    
    def _intent_aware_search(self, query: str, intents: Dict) -> List[Dict]:
        """Diagnostic intent-aware search"""
        print(f"\n🔍 Intent-Aware Search Debug")
        print(f"Primary Intent: {intents.get('response', {}).get('type')}")
        print(f"Query: {query}")
        
        # Get raw results
        results = self.collection.query(
            query_texts=[query],
            n_results=10,
            include=["metadatas", "distances", "documents"]
        )
        
        
        # Safely extract primary intent
        primary_intent = (intents.get('response', {})
                          .get('primary_intent', '')
                          .lower() if intents else '')
        
        print(f"\n🔍 Intent Analysis:")
        print(f"- Raw intents: {json.dumps(intents, indent=2)}")
        print(f"- Primary intent: {primary_intent}")
            
        return results["metadatas"][0]

    def _filter_by_intent(self, apis: List[Dict], intents: Dict) -> List[Dict]:
        """Priority sort based on detected intents"""
        return sorted(
            apis,
            key=lambda x: self._intent_match_score(x, intents),
            reverse=True
        )

    def _intent_match_score(self, api: Dict, intents: Dict) -> int:
        """Score API endpoints based on intent relevance"""
        score = 0
        if "supports_people_search" in api and "biographical" in intents.get("secondary_intents", []):
            score += 2
        if "supports_temporal_filtering" in api and "temporal" in intents.values():
            score += 1
        return score
    
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
    
    def _match_apis(self, query: str, entities: Dict, intents: Dict) -> List[Dict]:
        """Debug-enabled API matching"""
        print("\n=== MATCH_APIS DEBUG ===")
        print(f"Initial query: {query}")
        print(f"Current entities: {json.dumps(entities, indent=2)}")
        
        try:
            # Get raw candidates
            raw_results = self._get_raw_candidates(query, entities)
            print(f"\n🔍 Raw Chroma results: {raw_results.keys()}")
            
            if not raw_results or "metadatas" not in raw_results:
                print("⚠️ No results from ChromaDB query")
                return []
                
            # Parse metadata
            parsed_steps = self._parse_metadata(raw_results)
            print(f"\n🔄 Parsed {len(parsed_steps)} steps:")
            for idx, step in enumerate(parsed_steps[:3]):  # Print first 3 steps
                print(f"  {idx+1}. {step['path']} (score: {1 - step['distance']:.2f})")
            
            # Identify missing IDs
            missing_ids = self._get_missing_ids(parsed_steps, entities)
            print(f"\n❌ Missing IDs: {missing_ids}")
            
            # Prioritize steps
            prioritized = self._priority_sort(
                parsed_steps,
                needs_search=bool(missing_ids),
                missing_ids=missing_ids,
                intents=intents
            )
            
            print("\n🏆 Top 5 prioritized steps:")
            for idx, step in enumerate(prioritized[:5]):
                print(f"  {idx+1}. {step['path']} ({step['operation_type']})")
                
            return prioritized[:15]
            
        except Exception as e:
            print(f"\n🔥 MATCH_APIS ERROR: {str(e)}")
            traceback.print_exc()
            return []
    
    def _get_missing_ids(self, steps: List[Dict], entities: Dict) -> List[str]:
        """Identify unresolved ID requirements from planned steps"""
        required_ids = set()
        for step in steps:
            required_ids.update(step.get("resolution_deps", []))
        return [id for id in required_ids if id not in entities]
    
    def _get_raw_candidates(self, query: str, entities: Dict) -> Dict:
        """Debuggable candidate retrieval"""
        print("\n🔎 GET_RAW_CANDIDATES")
        print(f"Needs search? {not any(k.endswith('_id') for k in entities)}")
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=25,
                where={"requires_id_resolution": "False"} if not any(k.endswith('_id') for k in entities) else None,
                include=["metadatas", "distances", "documents"]
            )
            
            print(f"Chroma response keys: {results.keys()}")
            print(f"Received {len(results['metadatas'][0])} candidates")
            return results
            
        except Exception as e:
            print(f"\n🔥 GET_RAW_CANDIDATES ERROR: {str(e)}")
            traceback.print_exc()
            return {"metadatas": [], "distances": []}

    def _parse_metadata(self, raw_results: Dict) -> List[Dict]:
        """Debuggable metadata parsing"""
        print("\n🔧 PARSE_METADATA")
        parsed = []
        
        try:
            metas = raw_results.get("metadatas", [[]])[0] or []
            distances = raw_results.get("distances", [[]])[0] or []
            
            print(f"Processing {len(metas)} metadata entries")
            print(f"Sample metadata: {json.dumps(metas[0], indent=2) if metas else 'None'}")
            
            for meta, distance in zip(metas, distances):
                parsed.append({
                    "path": meta.get("path", "unknown"),
                    "method": meta.get("method", "GET"),
                    "operation_type": meta.get("operation_type", "unknown"),
                    "search_targets": MetadataParser.parse_list(meta.get("search_capable_for", "")),
                    "resolution_deps": MetadataParser.parse_list(meta.get("resolution_dependencies", "")),
                    "distance": distance,
                    "media_type": meta.get("media_type", "unknown")
                })
                
            return parsed
            
        except Exception as e:
            print(f"\n🔥 PARSE_METADATA ERROR: {str(e)}")
            traceback.print_exc()
            return []

    def _parse_metadata(self, raw_results: Dict) -> List[Dict]:
        """Convert ChromaDB metadata to application format"""
        parsed = []
    
    def _priority_sort(self, results: List[Dict], needs_search: bool) -> List[Dict]:
        """Context-aware ranking"""
        return sorted(
            results,
            key=lambda x: (
                -x['metadata']['search_capable_for'],  # Prioritize search
                not x['metadata']['requires_resolution'],  # Prefer no-ID endpoints first
                len(x['metadata']['resolution_dependencies'])  # Simpler first
            )
        )

    def _dynamic_priority_sort(self, metadatas: List[Dict], intents: Dict) -> List[Dict]:
        """Score results based on intent-relevant metadata"""
        scored = []
        
        for meta in metadatas:
            if not isinstance(meta, dict):
                continue
                
            score = 0
            
            # Context matching
            ctx = meta.get("context", "").lower()
            if "biographical" in intents.get("primary_intent", "").lower():
                if "person" in ctx or "people" in ctx:
                    score += 2
                if "search" in ctx:
                    score += 1
                    
            # Media type alignment
            if meta.get("media_type") == "person":
                score += 1.5
                
            # Parameter support
            if "person_id" in meta.get("parameters", {}).get("id_parameters", []):
                score += 1
                
            scored.append((meta, score))
        
        # Sort by score then distance
        return [x[0] for x in sorted(scored, key=lambda x: (-x[1], x[0].get("distance", 0)))]
    
    def _validate_plan(self, plan: List[Dict]) -> List[Dict]:
        """Flexible validation with field normalization"""
        valid_steps = []
        
        for step in plan:
            if not isinstance(step, dict):
                continue
                
            # Normalize field names
            endpoint = step.get("endpoint") or step.get("path")
            method = step.get("method") or step.get("http_method")
            
            if not endpoint or not method:
                print(f"⚠️ Invalid step: {json.dumps(step, indent=2)}")
                continue
                
            valid_steps.append({
                "endpoint": endpoint.strip(),
                "method": method.strip().upper(),
                "parameters": step.get("parameters", {})
            })
            
        return valid_steps

    def _intent_priority_score(self, api_meta: Dict, intents: Dict) -> float:
        score = 0.0
        if "supports_genre_filtering" in api_meta and "discovery" in intents:
            score += 0.5
        if "supports_temporal_filtering" in api_meta and "temporal" in intents:
            score += 0.7
        return score
    
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
    """
    Execute a TMDB API call with proper parameter substitution and error handling
    
    Args:
        api_call_info: Dictionary containing endpoint, method, and parameters
        entities: Resolved entities containing IDs for path parameter substitution
        
    Returns:
        Dictionary with endpoint as key and response/error as value
    """
    try:
        # Substitute path parameters from resolved entities
        substituted_endpoint = _replace_path_parameters(
            api_call_info["endpoint"],
            entities
        )
        
        # Make API request
        response = requests.request(
            method=api_call_info["method"].upper(),
            url=f"{BASE_URL}{substituted_endpoint}",
            headers=HEADERS,
            params=api_call_info.get("parameters", {})
        )
        
        return {
            "endpoint": api_call_info["endpoint"],
            "method": api_call_info["method"],
            "status": response.status_code,
            "data": response.json() if response.status_code == 200 else None,
            "error": None
        }
        
    except RequestException as req_err:
        return _handle_api_error(api_call_info["endpoint"], "HTTP error", str(req_err))
    except KeyError as key_err:
        return _handle_api_error(api_call_info["endpoint"], "Missing key", str(key_err))
    except Exception as general_err:
        return _handle_api_error(api_call_info["endpoint"], "Unexpected error", str(general_err))

def _replace_path_parameters(endpoint: str, entities: Dict) -> str:
    """Replace {parameter} placeholders in endpoint path with resolved entities"""
    return re.sub(
        r"{(\w+)}",
        lambda match: str(entities.get(f"{match.group(1)}_id", "")),
        endpoint
    )

def _handle_api_error(endpoint: str, error_type: str, message: str) -> Dict:
    """Generate consistent error response structure"""
    return {
        "endpoint": endpoint,
        "status": None,
        "data": None,
        "error": {
            "type": error_type,
            "message": message
        }
    }

def execute_planned_steps(planner_agent, query, extracted_entities):
    """
    Executes API calls dynamically while resolving dependencies.
    - If an API requires an ID, fetch it first via a query-based API.
    - Runs steps sequentially, updating shared_state dynamically.
    """

    execution_plan = planner_agent.generate_plan(
        query, 
        extracted_entities,
        {}  # Empty intents as fallback
    )
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
