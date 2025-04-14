from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Any
import json
import os
import chromadb
import re
from dotenv import load_dotenv
from nlp_retriever import (
    EnhancedIntentAnalyzer, 
    IntelligentPlanner
)
from llm_client import OpenAILLMClient
from entity_resolution import TMDBEntityResolver
from dependency_manager import ExecutionState, DependencyManager
from execution_orchestrator import ExecutionOrchestrator
from fallback_handler import FallbackHandler
from query_classifier import QueryClassifier
from param_resolver import ParamResolver
from prompt_templates import PROMPT_TEMPLATES, DEFAULT_TEMPLATE, PLAN_PROMPT



# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API configuration
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TMDB_API_KEY}"}

param_resolver = ParamResolver()
llm_client = OpenAILLMClient()
dependency_manager = DependencyManager()
query_classifier = QueryClassifier(api_key=OPENAI_API_KEY)

# Initialize core components
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

llm_client = OpenAILLMClient()
planner = IntelligentPlanner(
    chroma_collection=collection,
    param_resolver=param_resolver,
    llm_client=llm_client,
    dependency_manager=dependency_manager,
    query_classifier=query_classifier
)

class ControllerState(TypedDict):
    query: str
    execution_state: ExecutionState
    final_response: str
    api_context: str

def initialize_state(query: str) -> ControllerState:
    return {
        "query": query,
        "execution_state": ExecutionState(),
        "final_response": "",
        "api_context": ""
    }

# Node implementations
def parse_query(state: ControllerState) -> ControllerState:
    log_state_transition("parse", state)
    execution_state = state['execution_state']
    intent_analyzer = EnhancedIntentAnalyzer(llm_client)
    #query_classifier = QueryClassifier()
    
    # Extract entities and intents
    execution_state.raw_entities = intent_analyzer.extract_entities(state["query"])

    # Classify intents dynamically using the hybrid approach
    execution_state.detected_intents = classify_intent(state["query"])

    return state

def classify_intent(query):
    quick_result = query_classifier.classify(query)
    
    # Dynamically detect if deeper semantic analysis is needed
    if quick_result["primary_intent"] == "generic_search":
        intent_classifier = IntentClassifier(api_key=OPENAI_API_KEY)
        advanced_result = intent_classifier.classify(query)
        return advanced_result
    
    return quick_result

def resolve_entities(state: ControllerState) -> ControllerState:
    execution_state = state['execution_state']
    entity_resolver = TMDBEntityResolver(TMDB_API_KEY)
    
    print(f"\n{'='*30} ENTITY RESOLUTION {'='*30}")
    for ent_type in ["person", "movie", "tv", "genre"]:
        if ent_type in execution_state.raw_entities:
            print(f"🔍 Processing {ent_type} entity...")
            result = entity_resolver.resolve_entity(
                execution_state.raw_entities[ent_type][0], 
                ent_type
            )
            if result and 'id' in result:
                id_key = f"{ent_type}_id"
                execution_state.resolved_entities[id_key] = result['id']
                execution_state.track_entity_activity(
                    id_key, 'production', {'step_id': 'auto_resolve'}
                )
                print(f"✅ Resolved {id_key}: {result['id']}")
    return state

def plan_with_intent(state: ControllerState) -> ControllerState:
    execution_state = state['execution_state']
    dependency_manager = DependencyManager()
    
    try:
        # Use already-classified intents (DRY, no repeated classification)
        intent_context = execution_state.detected_intents
        primary_intent = intent_context.get('primary_intent', 'generic_search')
        secondary_intents = intent_context.get('secondary_intents', [])

        # Dynamically build an intent description for LLM prompt
        intent_description = f"Primary intent: {primary_intent}."
        if secondary_intents:
            intent_description += f" Secondary intents: {', '.join(secondary_intents)}."

        raw_plan = planner._llm_planning(  # Use pre-initialized planner
            prompt=PLAN_PROMPT.format(
                query=state["query"],
                entities=json.dumps(execution_state.resolved_entities),
                intents=intent_description,
                api_context=state["api_context"]
            ),
            dependencies=dependency_manager.graph
        )        

        # Validate plan structure
        if not isinstance(raw_plan, dict) or 'plan' not in raw_plan:
            raise ValueError("Invalid plan structure")

        execution_state.pending_steps = [
            step for step in raw_plan['plan']
            if _validate_step(step)
        ]

    except Exception as e:
        execution_state.error = f"Planning failed: {str(e)}"
    
    return state


def _validate_step(step: Dict) -> bool:
    required_keys = {'step_id', 'endpoint', 'method'}
    return all(key in step for key in required_keys)

def execute_api_plan(state: ControllerState) -> ControllerState:
    execution_state = state['execution_state']
    orchestrator = ExecutionOrchestrator(BASE_URL, HEADERS)
    fallback_handler = FallbackHandler()
    
    # Generate fallback if no steps
    if not execution_state.pending_steps:
        print("⚠️ No steps in plan - generating dynamic fallback")
        execution_state.pending_steps = fallback_handler.generate_steps(
            execution_state.resolved_entities,
            execution_state.detected_intents
        )
    
    # Dynamic path parameter injection
    for step in execution_state.pending_steps:
        # Extract path parameters from endpoint pattern
        path_params = re.findall(r"{(\w+)}", step.get("endpoint", ""))
        
        # Auto-inject entity references for path parameters
        step.setdefault("parameters", {})
        for param in path_params:
            if param in execution_state.resolved_entities:
                step["parameters"][param] = f"${param}"
            else:
                print(f"⚠️ Missing required path parameter: {param}")

    # Execute the plan with dynamic resolution
    updated_state = orchestrator.execute(execution_state)
    state['execution_state'] = updated_state
    
    return state

def retrieve_api_context(state: ControllerState) -> ControllerState:
    log_state_transition("retrieve_context", state)
    execution_state = state['execution_state']
    #planner = IntelligentPlanner(collection, ...)  # Your existing initialization
    #intent = execution_state.detected_intents.get('primary_intent') or "generic_search"


    # Get valid entity IDs only
    resolved_entities = {
        k: v for k, v in execution_state.resolved_entities.items() 
        if isinstance(v, (int, str))
    }
    
    try:
        state['api_context'] = planner.retriever.retrieve_context(
            query=state["query"],
            intent=execution_state.detected_intents.get('primary_intent', 'generic_search'),
            entities=execution_state.resolved_entities
        )
    except Exception as e:
        state["execution_state"].error = f"Context retrieval failed: {str(e)}"
    
    return state

def build_response(state: ControllerState) -> ControllerState:
    execution_state = state['execution_state']
    response = []
    
    # Extract intent type as string
    query_type = execution_state.detected_intents.get('primary', {}).get('type', 'generic')
    
    # Define formatters with string keys
    formatters = {
        'trending': _format_trending,
        'filmography': _format_filmography,
        'generic': _format_generic,
        'financial': _format_financial,
        'awards': _format_awards
    }
    
    # Get formatter using string key
    formatter = formatters.get(query_type, _format_generic)
    
    # Format response data
    for step_id, data in execution_state.data_registry.items():
        if data and 'data' in data:
            response.append(formatter(data['data']))
    
    state["final_response"] = "\n\n".join(response) or FallbackHandler.format_fallback(
        execution_state.resolved_entities
    )
    return state

# Response formatting
def _format_trending(data: List[Dict]) -> str:
    formatted = ["🎬 Currently Trending Movies:"]
    for idx, item in enumerate(data[:10]):
        formatted.append(
            f"{idx+1}. {item.get('title', 'Unknown')} "
            f"(⭐ {item.get('vote_average', 'N/A')})"
        )
    return "\n".join(formatted)

def _format_filmography(data: List[Dict]) -> str:
    formatted = ["🎥 Filmography:"]
    for idx, movie in enumerate(data[:10]):
        formatted.append(
            f"{idx+1}. {movie.get('title', 'Unknown')} "
            f"({movie.get('release_date', '')[:4]})"
        )
    return "\n".join(formatted)

def _format_generic(data: Dict) -> str:
    if 'name' in data:
        return f"ℹ️ {data['name']}: {data.get('biography', 'No information available')[:300]}..."
    return json.dumps(data, indent=2)

# Add missing formatters or remove references
def _format_financial(data: Dict) -> str:
    """Format financial data from TMDB API"""
    return f"""💰 Financial Details:
    Budget: ${data.get('budget', 0):,}
    Revenue: ${data.get('revenue', 0):,}
    Profit: ${data.get('revenue', 0) - data.get('budget', 0):,}"""

def _format_awards(data: Dict) -> str:
    """Format awards information"""
    awards = data.get('awards', [])
    if not awards:
        return "No awards information available"
    
    formatted = ["🏆 Awards:"]
    for award in awards[:5]:  # Show top 5 awards
        formatted.append(f"- {award.get('title')} ({award.get('year')})")
    return "\n".join(formatted)

def log_state_transition(node_name: str, state: ControllerState):
    print(f"\n🔄 State After {node_name.upper()} Node 🔄")
    print(f"| Query: {state['query']}")
    print(f"| API Context: {state['api_context'][:100] + '...' if state['api_context'] else '<empty>'}")
    print(f"| Resolved Entities: {state['execution_state'].resolved_entities}")
    print(f"| Pending Steps: {len(state['execution_state'].pending_steps)}")
    print(f"| Error: {state['execution_state'].error or '<none>'}")

# Update workflow construction with RAG retrieval
workflow = StateGraph(ControllerState)

# Define nodes (add retrieve_context node)
workflow.add_node("parse", parse_query)
workflow.add_node("resolve", resolve_entities)
workflow.add_node("retrieve_context", retrieve_api_context)  # New RAG node
workflow.add_node("plan", plan_with_intent)
workflow.add_node("execute", execute_api_plan)
workflow.add_node("respond", build_response)

# Configure edges with RAG sequence
workflow.set_entry_point("parse")
workflow.add_edge("parse", "resolve")
workflow.add_edge("resolve", "retrieve_context")  # Entity resolution first
workflow.add_edge("retrieve_context", "plan")     # Then RAG retrieval
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "respond")
workflow.add_edge("respond", END)

# Enhanced error handling with RAG context
def route_errors(state: ControllerState):
    """Handle both success and error cases with RAG awareness"""
    execution_state = state['execution_state']
    
    if execution_state.error:
        # If RAG context exists, try to use it for error recovery
        if 'api_context' in state and execution_state.pending_steps:
            return "execute"  # Retry with existing context
        return "respond"  # Final fallback to error response
    
    # Check for empty plan even after RAG
    if not execution_state.pending_steps and not execution_state.data_registry:
        return "retrieve_context"  # Try different retrieval strategy
    
    return END

# Add conditional edges for error recovery
workflow.add_conditional_edges(
    "execute",
    route_errors,
    {"respond": "respond", "retrieve_context": "retrieve_context", END: END}
)

# Add error path for RAG failures
workflow.add_conditional_edges(
    "retrieve_context",
    lambda s: "plan" if s.get('api_context') else "respond",
    {"plan": "plan", "respond": "respond"}
)

app = workflow.compile()

# Main handler
def handle_query(query: str) -> str:
    state = initialize_state(query)
    final_state = app.invoke(state)
    return final_state["final_response"]

if __name__ == "__main__":
    while True:
        query = input("\nEnter your media query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break
        print("\n=== RESPONSE ===")
        print(handle_query(query))