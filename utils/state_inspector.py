def print_state_snapshot(state: ExecutionState):
    print("\n=== STATE SNAPSHOT ===")
    print(f"Entities: {len(state['resolved_entities'])} resolved")
    print(f"Execution: {len(state['step_status'])} steps completed")
    print("Recent Activities:")
    for step, meta in list(state["step_status"].items())[-3:]:
        print(f"- {step}: {meta['status']} ({meta['duration']:.2f}s)")