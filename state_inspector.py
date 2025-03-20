def print_entity_lifecycle(state: ExecutionState):
    """Debug entity state transitions"""
    print("\n🔗 Entity Lifecycle:")
    for entity, usage in state['entity_usage'].items():
        source = state['entity_provenance'][entity].get('source')
        print(f"- {entity}:")
        print(f"  Source: {source}")
        print(f"  Used in: {', '.join(usage)}")
        print(f"  Current value: {state['resolved_entities'][entity]}")