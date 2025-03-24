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
    OpenAILLMClient
)
from entity_resolution import TMDBEntityResolver
from intent_classifier import IntentClassifier
from dependency_manager import ExecutionState, DependencyManager
import requests
from execution_orchestrator import ExecutionOrchestrator
from fallback_handler import FallbackHandler
from query_classifier import QueryClassifier
from param_resolver import ParamResolver
from prompt_templates import PROMPT_TEMPLATES, DEFAULT_TEMPLATE



# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API configuration
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TMDB_API_KEY}"}

param_resolver = ParamResolver()
llm_client = OpenAILLMClient(OPENAI_API_KEY)
dependency_manager = DependencyManager()
query_classifier = QueryClassifier()

# Initialize core components
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

class ControllerState(TypedDict):
    query: str
    execution_state: ExecutionState
    final_response: str

def initialize_state(query: str) -> ControllerState:
    return {
        "query": query,
        "execution_state": ExecutionState(),
        "final_response": ""
    }

# Node implementations
def parse_query(state: ControllerState) -> ControllerState:
    execution_state = state['execution_state']
    intent_analyzer = EnhancedIntentAnalyzer()
    query_classifier = QueryClassifier()
    
    # Extract entities and intents
    execution_state.raw_entities = intent_analyzer.extract_entities(state["query"])
    execution_state.detected_intents = {
        'primary': query_classifier.classify(state["query"])
    }
    return state

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

# def plan_with_intent(state: ControllerState) -> ControllerState:
#     execution_state = state['execution_state']
#     dependency_manager = DependencyManager()
#     # Initialize planner with correct parameters
#     planner = IntelligentPlanner(
#         chroma_collection=collection,
#         param_resolver=param_resolver,
#         llm_client=llm_client,
#         dependency_manager=dependency_manager,
#         query_classifier=query_classifier
#     )

#     # Sync resolved entities into the planner's registry
#     planner.entity_registry = execution_state.resolved_entities.copy()
    
#     try:
#         raw_plan = planner.generate_plan(
#             state["query"],
#             execution_state.resolved_entities,
#             execution_state.detected_intents
#         )
#         execution_state.pending_steps = [
#             step for step in raw_plan.get("plan", [])
#             if _should_retain_step(step, execution_state)
#         ]
#     except Exception as e:
#         execution_state.error = f"Planning failed: {str(e)}"
    
#     return state
def plan_with_intent(state: ControllerState) -> ControllerState:
    execution_state = state['execution_state']
    dependency_manager = DependencyManager()
    
    try:
        # Get dynamic classification
        classification = query_classifier.classify(state["query"])
        template_hint = PROMPT_TEMPLATES.get(
            classification["primary_intent"], 
            DEFAULT_TEMPLATE
        )

        # Build context with proper typing
        # context = {
        #     "query": state["query"],
        #     "entities": execution_state.resolved_entities,
        #     "intents": classification,
        #     "template_hint": template_hint
        # }

        planner = IntelligentPlanner(
            chroma_collection=collection,
            param_resolver=param_resolver,
            llm_client=llm_client,
            dependency_manager=dependency_manager,
            query_classifier=query_classifier
        )

        # Handle JSON parsing safely
        #raw_plan = planner.generate_plan(context)
        raw_plan = planner.generate_plan(
            state["query"],
            execution_state.resolved_entities,
            execution_state.detected_intents
        )

        # Validate plan structure
        if not isinstance(raw_plan, dict) or 'plan' not in raw_plan:
            raise ValueError("Invalid plan structure")

        execution_state.pending_steps = [
            step for step in raw_plan['plan']
            if _validate_step(step)  # Add your validation logic
        ]

    except Exception as e:
        execution_state.error = f"Planning failed: {str(e)}"
    
    return state

def _validate_step(step: Dict) -> bool:
    required_keys = {'step_id', 'endpoint', 'method'}
    return all(key in step for key in required_keys)

def _should_retain_step(step: Dict, state: ExecutionState) -> bool:
    """Dynamic step retention logic"""
    # Always keep data retrieval steps
    if step.get('operation_type') == 'data_retrieval':
        return True
        
    # Check entity dependencies
    required_entities = step.get('requires_entities', [])
    return not all(e in state.resolved_entities for e in required_entities)

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
    
    # Execute the plan
    updated_state = orchestrator.execute(execution_state)
    state['execution_state'] = updated_state
    return state

def build_response(state: ControllerState) -> ControllerState:
    execution_state = state['execution_state']
    response = []
    
    # Format based on query type
    query_type = execution_state.detected_intents.get('primary', 'generic')
    formatter = {
        'trending': _format_trending,
        'filmography': _format_filmography,
        # Remove or implement missing formatters
        'generic': _format_generic
    }.get(query_type, _format_generic)  # Default to generic
    
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

# Update workflow construction
workflow = StateGraph(ControllerState)

# Define nodes
workflow.add_node("parse", parse_query)
workflow.add_node("resolve", resolve_entities)
workflow.add_node("plan", plan_with_intent)
workflow.add_node("execute", execute_api_plan)
workflow.add_node("respond", build_response)

# Configure edges
workflow.set_entry_point("parse")
workflow.add_edge("parse", "resolve")
workflow.add_edge("resolve", "plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "respond")
workflow.add_edge("respond", END)  # Explicit END reference

# Error handling
def route_errors(state: ControllerState):
    """Handle both success and error cases"""
    if state['execution_state'].error:
        return "respond"  # Route errors to response builder
    return END

workflow.add_conditional_edges(
    "execute",
    route_errors,
    {"respond": "respond", END: END}  # Valid transitions
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