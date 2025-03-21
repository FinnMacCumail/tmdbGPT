from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Any, Optional
import json
import os
import chromadb
import re
import networkx as nx
import requests
import time
import traceback
from dotenv import load_dotenv
from utils.type_handling import convert_value

# Local imports
from nlp_retriever import (
    EnhancedIntentAnalyzer, 
    IntelligentPlanner,
    OpenAILLMClient
)
from entity_resolution import TMDBEntityResolver
from intent_classifier import IntentClassifier
from tmdbi_types import ExecutionState
from planning import classify_step, StepType
from state_inspector import (
    print_entity_lifecycle,
    print_state_snapshot,
    validate_state_consistency
)

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API configuration
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TMDB_API_KEY}"}

# ChromaDB initialization
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

class ControllerState(TypedDict):
    query: str
    raw_entities: Dict[str, List[str]]
    resolved_entities: Dict[str, Any]
    detected_intents: Dict[str, Any]
    api_plan: List[Dict[str, Any]]
    execution_results: Dict[str, Dict]
    dependency_graph: nx.DiGraph
    execution_state: Dict[str, Any]
    final_response: str

def initialize_state(query: str) -> ControllerState:
    """Initialize state with proper type hints and defaults"""
    return {
        "query": query,
        "raw_entities": {},
        "resolved_entities": {},
        "detected_intents": {},
        "api_plan": [],
        "execution_results": {},
        "dependency_graph": nx.DiGraph(),
        "execution_state": {
            "completed_steps": [],
            "available_entities": {},
            "step_output_types": {},
            "retry_count": 0,
            "critical_steps_executed": {},
            "emergency_resolutions": {}
        },
        "final_response": ""
    }

# Core components initialization
intent_analyzer = EnhancedIntentAnalyzer()
entity_resolver = TMDBEntityResolver(TMDB_API_KEY)
intent_classifier = IntentClassifier(OPENAI_API_KEY)
llm_client = OpenAILLMClient(OPENAI_API_KEY)
planner = IntelligentPlanner(collection, intent_analyzer, llm_client)

def parse_query(state: ControllerState) -> ControllerState:
    """Enhanced query parsing with state tracking"""
    state["raw_entities"] = intent_analyzer.extract_entities(state["query"])
    state["detected_intents"] = intent_classifier.classify(state["query"])
    return state

def resolve_entities(state: ControllerState) -> ControllerState:
    """Entity resolution with state instrumentation"""
    print(f"\n{'='*30} ENTITY RESOLUTION {'='*30}")
    state.setdefault("resolved_entities", {})
    
    for ent_type in ["person", "movie", "tv"]:
        if values := state["raw_entities"].get(ent_type):
            print(f"🔍 Resolving {ent_type}: {values[0]}")
            try:
                result = entity_resolver.resolve_entity(values[0], ent_type)
                if entity_id := result.get("id"):
                    id_key = f"{ent_type}_id"
                    state["resolved_entities"][id_key] = int(entity_id)
                    # Track entity provenance
                    state["execution_state"]["available_entities"][id_key] = {
                        "source": "entity_resolution",
                        "timestamp": time.time(),
                        "raw_value": values[0]
                    }
            except Exception as e:
                print(f"🚨 Resolution failed: {str(e)}")
                
    print("📦 Resolved Entities:")
    for k, v in state["resolved_entities"].items():
        print(f"- {k}: {v} ({type(v)})")
    
    return state

def plan_with_intent(state: ControllerState) -> ControllerState:
    """State-aware planning with validation"""
    print("\n=== INTENT-BASED PLANNING ===")
    try:
        plan_data = planner.generate_plan(
            state["query"],
            state["resolved_entities"],
            state["detected_intents"]
        )
        state["api_plan"] = plan_data["plan"]
        state["dependency_graph"] = plan_data["graph"]
        
        # Track plan generation in state
        state["execution_state"]["plan_generated"] = time.time()
        return state
    except Exception as e:
        print(f"⚠️ Planning failed: {str(e)}")
        state["api_plan"] = []
        return state

def execute_api_plan(state: ControllerState) -> ControllerState:
    """State-managed execution with critical step enforcement"""
    state.setdefault("execution_results", {})
    exec_state = state["execution_state"]
    
    print(f"\n{'='*30} EXECUTION PLAN {'='*30}")
    print_entity_lifecycle(state)
    
    try:
        dependency_graph = state["dependency_graph"]
        execution_order = list(nx.topological_sort(dependency_graph))
        
        print(f"\n🔀 Execution Order: {execution_order}")
        print("📊 Pre-Execution State:")
        print_state_snapshot(state)

        for step_id in execution_order:
            step_data = dependency_graph.nodes[step_id]
            print(f"\n🚀 Processing Step {step_id}: {step_data['description']}")
            
            # Critical step handling
            if step_data.get("metadata", {}).get("critical", False):
                print("⚡ Critical step enforcement")
                _handle_critical_step(step_data, state)

            # Parameter resolution
            resolved_params = _resolve_parameters(step_data, state)
            
            # API execution
            result = _execute_api_call(step_data, resolved_params)
            state["execution_results"][step_id] = result
            
            # Post-execution processing
            _process_execution_result(result, step_data, state)
            
            # Update execution state
            exec_state["completed_steps"].append(step_id)
            exec_state["step_output_types"][step_id] = (
                "entities" if step_data.get("output_entities") else "data"
            )

        print("\n✅ Execution Completed")
        print_state_snapshot(state)
        validate_state_consistency(state)
        return state

    except Exception as e:
        print(f"\n🚨 Execution Failed: {str(e)}")
        return _handle_execution_error(e, state)

def _handle_critical_step(step_data: Dict, state: ControllerState):
    """Enforce critical step execution"""
    exec_state = state["execution_state"]
    required_entities = re.findall(r"\$(\w+)", str(step_data["parameters"]))
    
    # Emergency entity resolution
    for entity in required_entities:
        if entity not in exec_state["available_entities"]:
            print(f"🚨 Emergency resolution for {entity}")
            entity_type = entity.split("_")[0]
            if raw_value := state["raw_entities"].get(entity_type, [None])[0]:
                exec_state["available_entities"][entity] = raw_value
                exec_state["emergency_resolutions"][entity] = time.time()
                print(f"⚡ Assigned raw value for {entity}: {raw_value}")

def _resolve_parameters(step_data: Dict, state: ControllerState) -> Dict:
    resolved_params = {}
    for param, value in step_data["parameters"].items():
        if isinstance(value, str) and value.startswith("$"):
            entity = value[1:]
            raw_value = state["execution_state"]["available_entities"].get(entity)
            
            # Get parameter type from endpoint schema
            param_type = _get_param_type(
                step_data["endpoint"], 
                param
            )
            
            # Convert value using unified system
            resolved_params[param] = convert_value(raw_value, param_type)
            
    return resolved_params
def _execute_api_call(step_data: Dict, params: Dict) -> Dict:
    """Execute API call with state tracking"""
    try:
        endpoint = step_data["endpoint"].format(**params)
        response = requests.request(
            method=step_data.get("method", "GET"),
            url=f"{BASE_URL}{endpoint}",
            headers=HEADERS,
            params=params
        )
        response.raise_for_status()
        return {"status": response.status_code, "data": response.json()}
    except Exception as e:
        return _handle_api_error(endpoint, str(e))

def _process_execution_result(result: Dict, step_data: Dict, state: ControllerState):
    """Handle API response and extract entities"""
    if result.get("data"):
        new_entities = _extract_entities(result["data"], step_data.get("output_entities", []))
        for entity, value in new_entities.items():
            state["execution_state"]["available_entities"][entity] = value
            print(f"📥 Stored entity: {entity} = {value}")

def _handle_execution_error(error: Exception, state: ControllerState) -> ControllerState:
    """Error handling with retry logic"""
    exec_state = state["execution_state"]
    exec_state["error"] = str(error)
    validate_state_consistency(state)
    
    if exec_state["retry_count"] < 3:
        exec_state["retry_count"] += 1
        print(f"🔄 Retry attempt {exec_state['retry_count']}/3")
        return execute_api_plan(state)
    
    print("❌ Maximum retries exceeded")
    return state

def build_response(state: ControllerState) -> ControllerState:
    """Generate final response with state validation"""
    print("\n=== RESPONSE GENERATION ===")
    if not validate_state_consistency(state):
        print("⚠️ Building response with validation errors")
    
    prompt = f"""
    Craft response for: {state['query']}
    Use these verified results:
    {json.dumps(state['execution_results'], indent=2)}
    """
    
    state["final_response"] = llm_client.generate_response(prompt)
    return state

# Workflow configuration
workflow = StateGraph(ControllerState)
workflow.add_node("parse", parse_query)
workflow.add_node("resolve", resolve_entities)
workflow.add_node("plan", plan_with_intent)
workflow.add_node("execute", execute_api_plan)
workflow.add_node("respond", build_response)

workflow.set_entry_point("parse")
workflow.add_edge("parse", "resolve")
workflow.add_edge("resolve", "plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "respond")
workflow.add_edge("respond", END)

# Error handling
workflow.add_conditional_edges(
    "execute",
    lambda s: "respond" if s["execution_state"].get("error") else END,
    {"respond": "respond"}
)

app = workflow.compile()

def handle_query(query: str) -> str:
    """Public query interface"""
    state = initialize_state(query)
    final_state = app.invoke(state)
    return final_state["final_response"]

if __name__ == "__main__":
    while True:
        try:
            query = input("\nEnter media query (or 'exit'): ")
            if query.lower() in ["exit", "quit"]:
                break
            print(f"\n{handle_query(query)}")
        except KeyboardInterrupt:
            break