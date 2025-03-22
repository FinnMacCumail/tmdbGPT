from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Any
import json
import os
import chromadb
import re
from dotenv import load_dotenv
from nlp_retriever import (
    EnhancedIntentAnalyzer, 
    IntelligentPlanner,
    execute_api_call,
    OpenAILLMClient
)
from entity_resolution import TMDBEntityResolver
from intent_classifier import IntentClassifier
from dependency_manager import ExecutionState, DependencyManager
import networkx as nx
import traceback
import requests
from execution_orchestrator import ExecutionOrchestrator

# Load environment variables first
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# TMDB API configuration
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TMDB_API_KEY}"}

# Initialize ChromaDB client with proper configuration
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

# Initialize core components
intent_analyzer = EnhancedIntentAnalyzer()
entity_resolver = TMDBEntityResolver(TMDB_API_KEY)
intent_classifier = IntentClassifier(OPENAI_API_KEY)
llm_client = OpenAILLMClient(OPENAI_API_KEY) 
planner = IntelligentPlanner(collection, intent_analyzer, llm_client)
dependency_manager = DependencyManager()
execution_orchestrator = ExecutionOrchestrator(BASE_URL, HEADERS)

class ControllerState(TypedDict):
    query: str
    execution_state: ExecutionState
    raw_entities: Dict
    detected_intents: Dict
    final_response: str

def initialize_state(query: str) -> ControllerState:
    return {
        "query": query,
        "raw_entities": {},
        "detected_intents": {},
        "execution_state": ExecutionState(),
        "final_response": ""
    }

# Define node functions
def parse_query(state: ControllerState) -> ControllerState:
    """Parse query and store in execution state"""
    execution_state = state['execution_state']
    
    execution_state.raw_entities = intent_analyzer.extract_entities(state["query"])
    execution_state.detected_intents = intent_classifier.classify(state["query"])
    
    return state

def resolve_entities(state: ControllerState) -> ControllerState:
    """Entity resolution with execution state integration"""
    execution_state = state['execution_state']
    
    print(f"\n{'='*30} ENTITY RESOLUTION {'='*30}")
    for ent_type in ["person", "movie", "tv"]:
        if ent_type in execution_state.raw_entities:
            print(f"\n🔍 Processing {ent_type} entity...")
            result = entity_resolver.resolve_entity(
                execution_state.raw_entities[ent_type][0], 
                ent_type
            )
            
            if result and 'id' in result:
                id_key = f"{ent_type}_id"
                execution_state.resolved_entities[id_key] = result['id']
                execution_state.track_entity_activity(
                    id_key, 
                    'production', 
                    {'step_id': 'entity_resolution', 'type': 'auto'}
                )
                print(f"✅ Resolved {id_key}: {result['id']}")

    return state


#Updated planning node
def plan_with_intent(state: ControllerState) -> ControllerState:
    """State-integrated planning"""
    execution_state = state['execution_state']
    
    print("\n=== INTENT-BASED PLANNING ===")
    try:
        full_plan = planner.generate_plan(
            state["query"], 
            execution_state.resolved_entities,
            execution_state.detected_intents
        )
        
        # Update execution state
        execution_state.pending_steps = [
            step for step in full_plan["plan"]
            if not _should_skip_step(step, execution_state)
        ]
        
        # Build dependency graph
        dependency_manager.analyze_dependencies(execution_state.pending_steps)
        execution_state.dependency_graph = dependency_manager.execution_state.dependency_graph
        
        print(f"\n📋 Execution Plan Ready: {len(execution_state.pending_steps)} steps")
        return state
        
    except Exception as e:
        print(f"Planning error: {str(e)}")
        execution_state.pending_steps = []
        return state

# Helper functions
def _should_skip_step(step: Dict, state: ExecutionState) -> bool:
    """Enhanced skip logic considering data steps"""
    # Never skip data retrieval steps
    if step.get('operation_type') == 'data_retrieval':
        return False
        
    # Only skip if ALL outputs exist
    outputs = step.get('output_entities', [])
    return all(e in state.resolved_entities for e in outputs)

def execute_api_plan(state: ControllerState) -> ControllerState:
    """Execute using enhanced orchestration"""
    execution_state = state['execution_state']
    
    print(f"\n{'='*30} EXECUTION PLAN {'='*30}")
    
    try:
        state['execution_state'] = execution_orchestrator.execute_plan(execution_state)
    except Exception as e:
        # Use proper error field assignment
        state['execution_state'].error = f"Critical execution error: {str(e)}"
    
    return state

def get_param_type(self, endpoint: str, param: str) -> str:
    """Dynamic parameter type lookup"""
    result = self.collection.query(
        query_texts=[endpoint],
        n_results=1,
        where={"method": "GET"},
        include=["metadatas"]
    )
    
    if result["metadatas"][0]:
        params = json.loads(result["metadatas"][0][0].get("parameters", "[]"))
        for p in params:
            if p["name"] == param:
                return p.get("schema", {}).get("type", "string")
    return "string"

def convert_type(self, value: Any, target_type: str) -> Any:
    """Type conversion with error handling"""
    try:
        if target_type == "integer":
            return int(value)
        elif target_type == "boolean":
            return bool(value)
        elif target_type == "number":
            return float(value)
        return str(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert {value} to {target_type}: {str(e)}")
    
def get_required_entities(self, step_data: Dict) -> List[str]:
    """Extract entity references from parameters"""
    entities = []
    for param, value in step_data.get('parameters', {}).items():
        if isinstance(value, str) and value.startswith("$"):
            entities.append(value[1:])
    return entities

def resolve_missing_entities(self, entities: List[str], state: ControllerState):
    """Dynamic entity resolution fallback"""
    for entity in entities:
        print(f"🔎 Attempting to resolve {entity}...")
        # Extract entity type from naming convention
        entity_type = entity.split('_')[0]  # person_id -> person
        raw_value = state['raw_entities'].get(entity_type, [None])[0]
        
        if raw_value:
            resolved = self.entity_resolver.resolve_entity(raw_value, entity_type)
            if resolved and 'id' in resolved:
                state['execution_state']['available_entities'][entity] = resolved['id']
                print(f"✅ Resolved {entity} = {resolved['id']}")

def _extract_entities_from_response(data: Dict, output_entities: List[str]) -> Dict:
    """Entity extraction with detailed debugging"""
    entities = {}
    print(f"\n🔍 Entity Extraction from Response:")
    
    # Handle paginated results
    if 'results' in data:
        print(f"Processing results array ({len(data['results'])} items)")
        if data['results']:
            first_item = data['results'][0]
            print(f"First item keys: {list(first_item.keys())}")
            for entity in output_entities:
                if entity.endswith('_id') and 'id' in first_item:
                    entities[entity] = first_item['id']
                elif entity in first_item:
                    entities[entity] = first_item[entity]
    
    # Handle single entity responses
    elif 'id' in data:
        print("Processing single entity response")
        entity_type = data.get('media_type', 'unknown')
        for entity in output_entities:
            if entity == f"{entity_type}_id":
                entities[entity] = data['id']
            elif entity in data:
                entities[entity] = data[entity]
    
    # Debug output
    if entities:
        print("Extracted Entities:")
        for k, v in entities.items():
            print(f"- {k}: {v}")
    else:
        print("⚠️ No entities extracted from response")
        print(f"Response keys: {list(data.keys())}")
        
    return entities

def _extract_entities_from_response(data: Dict, output_entities: List[str]) -> Dict:
    """Extract entities from API response with debugging"""
    entities = {}
    
    print(f"🔎 Extracting entities: {output_entities}")
    
    # Handle paginated results
    if 'results' in data:
        print(f"📄 Processing results array ({len(data['results'])} items)")
        if data['results']:
            first_item = data['results'][0]
            for entity in output_entities:
                if entity.endswith('_id') and 'id' in first_item:
                    entities[entity] = first_item['id']
                    print(f"✅ Extracted {entity} = {first_item['id']}")
    
    # Handle single entity responses
    elif 'id' in data:
        print("📄 Processing single entity response")
        entity_type = data.get('media_type', 'unknown')
        base_entity = f"{entity_type}_id"
        for entity in output_entities:
            if entity == base_entity:
                entities[entity] = data['id']
                print(f"✅ Extracted {entity} = {data['id']}")
    
    # Debug case where no entities found
    if not entities:
        print("⚠️ No entities extracted from response")
        print(f"Response data: {json.dumps(data, indent=2)[:500]}...")
    
    return entities


def build_response(state: ControllerState) -> ControllerState:
    """Build response with fallback handling"""
    execution_state = state['execution_state']
    
    print("\n=== BUILDING FINAL RESPONSE ===")
    
    if execution_state.data_registry:
        state["final_response"] = _format_api_response(execution_state.data_registry)
    elif execution_state.resolved_entities:
        state["final_response"] = _format_entity_fallback(execution_state.resolved_entities)
    else:
        state["final_response"] = "No relevant information could be found."
    
    return state

def _resolve_step_parameters(step: Dict, entities: Dict) -> Dict:
    """Resolve parameters using execution state"""
    resolved = {}
    for param, value in step.get('parameters', {}).items():
        if isinstance(value, str) and value.startswith("$"):
            entity_key = value[1:]
            resolved[param] = entities.get(entity_key, value)
        else:
            resolved[param] = value
    return resolved

def _execute_api_step(step: Dict, params: Dict) -> Dict:
    """Execute single API step"""
    try:
        response = requests.request(
            method=step.get('method', 'GET'),
            url=f"{BASE_URL}{step['endpoint']}",
            headers=HEADERS,
            params=params
        )
        return {
            "status": response.status_code,
            "data": response.json() if response.ok else None,
            "error": None
        }
    except Exception as e:
        return {"status": None, "data": None, "error": str(e)}
    
def _format_api_response(data: Dict) -> str:
    """Enhanced response formatting"""
    responses = []
    for step_id, result in data.items():
        if result.get('data'):
            responses.append(json.dumps(result['data'], indent=2))
    return "\n\n".join(responses) if responses else "No API data available"

def _format_entity_fallback(entities: Dict) -> str:
    """Improved entity-based fallback"""
    return "\n".join([
        f"{k.replace('_', ' ').title()}: {v}"
        for k, v in entities.items()
        if v is not None
    ]) or "No entity information available"

# Updated workflow construction
workflow = StateGraph(ControllerState)

# Add nodes with updated names
workflow.add_node("parse", parse_query)
workflow.add_node("resolve", resolve_entities)
workflow.add_node("plan", plan_with_intent)
workflow.add_node("execute", execute_api_plan)
workflow.add_node("respond", build_response)

# Set up edges
workflow.set_entry_point("parse")
workflow.add_edge("parse", "resolve")
workflow.add_edge("resolve", "plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "respond")
workflow.add_edge("respond", END)

# Conditional error handling
def route_errors(state: ControllerState):
    if state['execution_state'].error:
        return "handle_error"
    return END

workflow.add_conditional_edges(
    "execute",
    route_errors,
    {"handle_error": "respond", END: END}
)

app = workflow.compile()

# Main application handler
def handle_query(query: str) -> str:
    """Public interface for the application"""
    state = initialize_state(query)
    final_state = app.invoke(state)
    return final_state["final_response"]

if __name__ == "__main__":
    while True:
        query = input("\nEnter your media query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break
        response = handle_query(query)
        print("\n=== FINAL RESPONSE ===")
        print(response)