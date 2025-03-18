from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List
from nlp_retriever import EnhancedIntentAnalyzer, IntelligentPlanner, execute_planned_steps
from entity_resolution import TMDBEntityResolver
from semantic_embed import GENRE_MAPPINGS
import json
import os

# Initialize core components
intent_analyzer = EnhancedIntentAnalyzer()
entity_resolver = TMDBEntityResolver(os.getenv("TMDB_API_KEY"))
planner = IntelligentPlanner(collection, intent_analyzer)

class ControllerState(TypedDict):
    query: str
    raw_entities: Dict
    resolved_entities: Dict
    api_plan: List[Dict]
    execution_results: Dict
    final_response: str

def initialize_state(query: str) -> ControllerState:
    return {
        "query": query,
        "raw_entities": {},
        "resolved_entities": {},
        "api_plan": [],
        "execution_results": {},
        "final_response": ""
    }

# Define node functions
def parse_query(state: ControllerState) -> ControllerState:
    """Node 1: Parse query and extract raw entities"""
    print("\n=== PARSING QUERY ===")
    state["raw_entities"] = intent_analyzer.extract_entities(state["query"])
    return state

def resolve_entities(state: ControllerState) -> ControllerState:
    """Node 2: Resolve entities using TMDB APIs"""
    print("\n=== RESOLVING ENTITIES ===")
    resolved = {}
    
    # Resolve different entity types
    for ent_type, values in state["raw_entities"].items():
        if not values: continue
        
        for value in values:
            # Handle special cases first
            if ent_type == "genre":
                resolved_genre = entity_resolver.resolve_genre(value)
                if resolved_genre:
                    resolved.setdefault("genres", []).append(resolved_genre["id"])
            elif ent_type == "person":
                person = entity_resolver.resolve_person(value)
                if person:
                    resolved["person_id"] = person["id"]
            elif ent_type == "year":
                resolved["year"] = value
            # Add other entity types as needed
    
    state["resolved_entities"] = resolved
    print(f"Resolved Entities: {json.dumps(resolved, indent=2)}")
    return state

def plan_steps(state: ControllerState) -> ControllerState:
    """Node 3: Generate API execution plan"""
    print("\n=== GENERATING EXECUTION PLAN ===")
    plan = planner.generate_plan(state["query"], state["raw_entities"])
    state["api_plan"] = plan.get("plan", [])
    print(f"Execution Plan:\n{json.dumps(plan, indent=2)}")
    return state

def execute_api_plan(state: ControllerState) -> ControllerState:
    """Node 4: Execute the API plan"""
    print("\n=== EXECUTING API PLAN ===")
    results = {}
    
    for step in state["api_plan"]:
        if step["type"] == "api_call":
            print(f"Executing: {step['endpoint']}")
            response = execute_api_call({
                "endpoint": step["endpoint"],
                "method": step["method"],
                "parameters": step["parameters"]
            }, state["resolved_entities"])
            
            results[step["endpoint"]] = response
            # Store IDs for subsequent steps
            if "id" in response:
                state["resolved_entities"]["id"] = response["id"]
    
    state["execution_results"] = results
    return state

def build_response(state: ControllerState) -> ControllerState:
    """Node 5: Build final response"""
    print("\n=== BUILDING FINAL RESPONSE ===")
    llm_client = OpenAILLMClient(OPENAI_API_KEY)
    
    prompt = f"""
    Combine the following API results into a natural language response for the query:
    Query: {state['query']}
    
    API Results:
    {json.dumps(state['execution_results'], indent=2)}
    
    Provide a concise, human-readable answer structured as follows:
    1. Directly address the user's question
    2. Present key information from the API responses
    3. Cite sources where appropriate
    """
    
    state["final_response"] = llm_client.generate_response(prompt)
    return state

# Create workflow graph
workflow = StateGraph(ControllerState)

# Add nodes
workflow.add_node("parse", parse_query)
workflow.add_node("resolve", resolve_entities)
workflow.add_node("plan", plan_steps)
workflow.add_node("execute", execute_api_plan)
workflow.add_node("respond", build_response)

# Define edges
workflow.set_entry_point("parse")
workflow.add_edge("parse", "resolve")
workflow.add_edge("resolve", "plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "respond")
workflow.add_edge("respond", END)

# Add conditional edges for error handling
def handle_errors(state: ControllerState) -> str:
    if "error" in state.get("execution_results", {}):
        return "error_handler"
    return END

workflow.add_conditional_edges(
    "execute",
    handle_errors,
    {"error_handler": "respond", END: END}
)

# Compile the graph
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