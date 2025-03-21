def print_entity_lifecycle(state: ExecutionState):
    """Debug entity state transitions"""
    print("\n🔗 Entity Lifecycle:")
    for entity, usage in state['entity_usage'].items():
        source = state['entity_provenance'][entity].get('source')
        print(f"- {entity}:")
        print(f"  Source: {source}")
        print(f"  Used in: {', '.join(usage)}")
        print(f"  Current value: {state['resolved_entities'][entity]}")

def validate_state_consistency(state: ExecutionState):
    """Ensure entity declarations match usage patterns"""
    errors = []
    
    # Check for unused resolved entities
    for entity in state["resolved_entities"]:
        if entity not in state["entity_usage"]:
            errors.append(f"Unused entity: {entity}")
    
    # Verify step outputs match declarations
    for step_id, output_type in state["step_output_types"].items():
        if output_type == "entities" and not state["plan"][step_id].get("output_entities"):
            errors.append(f"Step {step_id} claims entity output but declares none")
    
    if errors:
        print("\n🚨 State Validation Errors:")
        for error in errors:
            print(f"- {error}")
        return False
    return True