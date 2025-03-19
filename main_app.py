from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List
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

# Load environment variables first
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB client with proper configuration
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

# Verify collection contents
print(f"Collection contains {collection.count()} embeddings")

# Initialize core components
intent_analyzer = EnhancedIntentAnalyzer()
entity_resolver = TMDBEntityResolver(TMDB_API_KEY)
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
    raw_entities = state["raw_entities"]
    resolved = state["resolved_entities"]
    
    print("\n=== ENTITY RESOLUTION DEBUG ===")
    print(f"🐞 [ENTITY RESOLVER] Raw entities received: {json.dumps(raw_entities, indent=2)}")
    
    for ent_type in ["person", "movie", "tv"]:
        if ent_type in raw_entities and raw_entities[ent_type]:
            raw_value = raw_entities[ent_type][0]
            query_key = f"{ent_type}_query"
            id_key = f"{ent_type}_id"
            
            print(f"\n🐞 [ENTITY RESOLVER] Processing {ent_type} entity")
            print(f"🐞 [ENTITY RESOLVER] Raw value: {raw_value}")
            
            # Store query value
            resolved[query_key] = raw_value
            print(f"✅ [ENTITY RESOLVER] Stored query value: {query_key} = {raw_value}")

            # Attempt ID resolution
            try:
                print(f"🐞 [ENTITY RESOLVER] Attempting {ent_type} ID resolution...")
                result = entity_resolver.resolve_entity(raw_value, ent_type)
                
                if result and "id" in result:
                    resolved[id_key] = result["id"]
                    print(f"✅ [ENTITY RESOLVER] Resolved {ent_type} ID: {result['id']}")
                else:
                    print(f"⚠️ [ENTITY RESOLVER] No ID found for {ent_type}: {raw_value}")
                    
            except Exception as e:
                print(f"🔥 [ENTITY RESOLVER] Error resolving {ent_type}: {str(e)}")
                #print(f"🔥 [ENTITY RESOLVER] Traceback: {traceback.format_exc()}")

    print(f"\n🐞 [ENTITY RESOLVER] Final resolved entities: {json.dumps(resolved, indent=2)}")
    return state

def plan_steps(state: ControllerState) -> ControllerState:
    """Node 3: Generate API execution plan with validation"""
    print("\n=== GENERATING EXECUTION PLAN ===")
    try:
        plan = planner.generate_plan(state["query"], state["resolved_entities"])
        # Ensure all steps have required fields
        valid_steps = [
            step for step in plan.get("plan", [])
            if "endpoint" in step and "method" in step
        ]
        state["api_plan"] = valid_steps
        print(f"Execution Plan:\n{json.dumps(valid_steps, indent=2)}")
    except Exception as e:
        print(f"⚠️ Planning Failed: {str(e)}")
        state["api_plan"] = []
    return state

def execute_api_plan(state: ControllerState) -> ControllerState:
    """Execution with detailed request/response logging"""
    print("\n=== EXECUTION DEBUG ===")
    results = {}
    
    for step in state["api_plan"]:
        print(f"\n🚀 Executing step: {step['endpoint']}")
        print(f"🔧 Method: {step['method']}")
        print(f"🔧 Parameters: {json.dumps(step.get('parameters', {}), indent=2)}")
        
        try:
            response = execute_api_call(step, state["resolved_entities"])
            print(f"✅ Response received (status: {response.get('status_code', 'N/A')})")
            
            # Store relevant IDs from responses
            if "id" in response:
                entity_type = step["endpoint"].split("/")[1]
                id_key = f"{entity_type}_id"
                state["resolved_entities"][id_key] = response["id"]
                print(f"💡 Stored resolved {id_key} = {response['id']}")
            
            results[step["endpoint"]] = response
            
        except Exception as e:
            print(f"🚨 Execution failed: {str(e)}")
            results[step["endpoint"]] = {"error": str(e)}
    
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