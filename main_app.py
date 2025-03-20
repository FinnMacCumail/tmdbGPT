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
from dependency_manager import DependencyManager, ExecutionState
import networkx as nx

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
#print(f"Collection contains {collection.count()} embeddings")

# Initialize core components
intent_analyzer = EnhancedIntentAnalyzer()
entity_resolver = TMDBEntityResolver(TMDB_API_KEY)
intent_classifier = IntentClassifier(OPENAI_API_KEY)
llm_client = OpenAILLMClient(OPENAI_API_KEY) 
planner = IntelligentPlanner(collection, intent_analyzer, llm_client)

class ControllerState(TypedDict):
    query: str
    raw_entities: Dict
    resolved_entities: Dict
    detected_intents: Dict
    api_plan: Dict  
    execution_results: Dict
    dependency_graph: Any  # NetworkX graph
    execution_state: Dict
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
    """Enhanced parsing with intent classification"""
    #print("\n=== PARSING QUERY ===")
    state["raw_entities"] = intent_analyzer.extract_entities(state["query"])
    state["detected_intents"] = intent_classifier.classify(state["query"])
    return state

def resolve_entities(state: ControllerState) -> ControllerState:
    raw_entities = state["raw_entities"]
    resolved = state["resolved_entities"]
    
    #print("\n=== ENTITY RESOLUTION DEBUG ===")
    #print(f"🐞 [ENTITY RESOLVER] Raw entities received: {json.dumps(raw_entities, indent=2)}")
    
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
    planner.resolved_entities = resolved
    return state

# Add new node for intent-aware planning
def plan_with_intent(state: ControllerState) -> ControllerState:
    """Diagnostic planning node"""
    print("\n=== INTENT-BASED PLANNING ===")
    try:
        # Generate full plan structure
        full_plan = planner.generate_plan(
            state["query"], 
            state["resolved_entities"],
            state.get("detected_intents", {})
        )
        
        # Store both plan and graph
        state["api_plan"] = full_plan["plan"]
        state["dependency_graph"] = planner.execution_graph
        
        print("\n🔗 Dependency Graph Structure:")
        print(f"Nodes: {state['dependency_graph'].nodes(data=True)}")
        print(f"Edges: {state['dependency_graph'].edges()}")
        
        return state
        
    except Exception as e:
        print(f"Planning error: {str(e)}")
        state["api_plan"] = []
        state["dependency_graph"] = nx.DiGraph()  # Empty graph as fallback
        return state
         
def execute_api_plan(state: ControllerState) -> ControllerState:
    """Execute API plan with detailed step debugging and dependency resolution"""
    print(f"\n{'='*30} EXECUTION PLAN {'='*30}")
    
    # Initialize execution state
    state.setdefault('execution_results', {})
    state.setdefault('execution_state', {
        'completed_steps': [],
        'available_entities': state.get('resolved_entities', {}).copy()
    })
    
    # Validate dependency graph
    dependency_graph = state.get('dependency_graph', nx.DiGraph())
    print(f"\n🔗 Dependency Graph Structure:")
    print(f"Nodes ({len(dependency_graph.nodes)}): {list(dependency_graph.nodes)}")
    print(f"Edges ({len(dependency_graph.edges)}): {list(dependency_graph.edges)}")
    
    try:
        # Get execution order with cycle handling
        try:
            execution_order = list(nx.topological_sort(dependency_graph))
        except nx.NetworkXUnfeasible:
            print("⚠️ Circular dependencies detected, using insertion order")
            execution_order = list(dependency_graph.nodes)
            
        print(f"\n🔀 Execution Order: {execution_order}")

        for step_id in execution_order:
            step_data = dependency_graph.nodes[step_id]
            print(f"\n{'='*25} STEP {step_id} {'='*25}")
            print(f"📝 Description: {step_data.get('description', 'No description')}")
            print(f"🌐 Endpoint: {step_data['validated_endpoint']}")
            print(f"⚙️ Method: {step_data.get('method', 'GET')}")

            # Resolve parameters with entity tracking
            resolved_params = {}
            print(f"\n🔧 Parameter Resolution:")
            print(f"Available Entities: {json.dumps(state['execution_state']['available_entities'], indent=2)}")
            
            for param, value in step_data.get('parameters', {}).items():
                original_value = value
                if isinstance(value, str) and value.startswith("$"):
                    entity_key = value[1:]
                    resolved_value = state['execution_state']['available_entities'].get(entity_key)
                    print(f"🔎 Resolving {value}: ", end="")
                    
                    if resolved_value:
                        print(f"Found {resolved_value}")
                        resolved_params[param] = resolved_value
                    else:
                        print(f"Missing! Attempting fuzzy search...")
                        resolved_value = entity_resolver.fuzzy_search(entity_key, param.split('_')[0])
                        if resolved_value:
                            print(f"✅ Resolved via fuzzy search: {resolved_value['id']}")
                            resolved_params[param] = resolved_value['id']
                            state['execution_state']['available_entities'][entity_key] = resolved_value['id']
                        else:
                            raise ValueError(f"Missing required entity: {entity_key}")
                else:
                    resolved_params[param] = value
                    print(f"✅ Literal parameter: {param}={value}")

            # Handle path parameters
            path = step_data['validated_endpoint']
            for match in re.finditer(r'{(\w+)}', path):
                param_name = match.group(1)
                if param_name not in resolved_params:
                    raise ValueError(f"Missing path parameter: {param_name}")
                path = path.replace(f'{{{param_name}}}', str(resolved_params[param_name]))
            
            print(f"\n🚀 Final Request Details:")
            print(f"Resolved Path: {path}")
            print(f"Query Parameters: {json.dumps(resolved_params, indent=2)}")

            # Execute API call
            try:
                response = requests.request(
                    method=step_data.get('method', 'GET'),
                    url=f"{BASE_URL}{path}",
                    headers=HEADERS,
                    params=resolved_params
                )
                response.raise_for_status()
                
                result = {
                    "status": response.status_code,
                    "data": response.json(),
                    "error": None
                }
                
                print(f"\n✅ Success Response ({response.status_code}):")
                print(json.dumps(result['data'], indent=2)[:500] + ("..." if len(result['data']) > 500 else ""))
                
            except Exception as e:
                result = _handle_api_error(path, "Execution error", str(e))
                print(f"\n❌ API Call Failed:")
                print(f"Error: {str(e)}")
                print(f"Request Details:")
                print(f"- URL: {path}")
                print(f"- Params: {json.dumps(resolved_params, indent=2)}")

            # Store results and extract entities
            state['execution_results'][step_id] = result
            state['execution_state']['completed_steps'].append(step_id)
            
            if result['data']:
                new_entities = _extract_entities_from_response(
                    result['data'],
                    step_data.get('output_entities', [])
                )
                print(f"\n📥 Extracted Entities:")
                for k, v in new_entities.items():
                    print(f"- {k}: {v}")
                    state['execution_state']['available_entities'][k] = v

        return state

    except Exception as e:
        print(f"\n🚨 Execution Failed: {str(e)}")
        traceback.print_exc()
        state['execution_state']['error'] = str(e)
        return state

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
workflow.add_node("intent_plan", plan_with_intent)  # New node
workflow.add_node("execute", execute_api_plan)
workflow.add_node("respond", build_response)

# Define edges
workflow.set_entry_point("parse")
workflow.add_edge("parse", "resolve")
workflow.add_edge("resolve", "intent_plan")
workflow.add_edge("intent_plan", "execute")
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