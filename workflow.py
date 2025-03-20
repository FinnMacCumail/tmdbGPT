from langgraph.graph import StateGraph, END
from tmdbi_types import ExecutionState

def create_core_workflow():
    workflow = StateGraph(ExecutionState)
    
    # Define Nodes
    workflow.add_node("resolve_entities", entity_resolution_node)
    workflow.add_node("generate_plan", planning_node)
    workflow.add_node("validate_steps", validation_node)
    workflow.add_node("execute_steps", execution_node)
    workflow.add_node("build_response", response_node)
    
    # Primary Flow
    workflow.set_entry_point("resolve_entities")
    workflow.add_edge("resolve_entities", "generate_plan")
    workflow.add_edge(
        "resolve_entities", 
        "generate_plan",
        condition=lambda s: _validate_state_shape(s) and s["resolved_entities"]
    )
    
    # Conditional Validation
    workflow.add_conditional_edges(
        "generate_plan",
        should_validate_plan,
        {"validate": "validate_steps", "skip": "execute_steps"}
    )
    
    workflow.add_edge("validate_steps", "execute_steps")
    workflow.add_edge("execute_steps", "build_response")
    workflow.add_edge("build_response", END)
    
    return workflow

def should_validate_plan(state: ExecutionState):
    return len(state['validation_errors']) > 0

# Add state shape validation at edges
def _validate_state_shape(state: ExecutionState):
    required_keys = {"query", "resolved_entities", "entity_origins"}
    if not required_keys.issubset(state.keys()):
        raise ValueError(f"Missing critical state keys: {required_keys - state.keys()}")