def log_summary(state, header=None):
    import logging
    from pprint import pformat

    logger = logging.getLogger("tmdbgpt-summary")
    print = logger.info  # Replace print with logger for structured logs

    if header:
        print(header)

    print("🧠 SUMMARY REPORT")
    print("=" * 60)

    # Input
    query = getattr(state, 'query', 'N/A')
    print(f"🔹 Query: {query}")

    # Media type
    media_type = getattr(state, 'intended_media_type', 'N/A')
    print(f"🎥 Media Type: {media_type}")

    # Extracted Entities
    ents = getattr(state, "extraction_result", {}).get("query_entities", [])
    if ents:
        print("🧾 Extracted Entities:")
        for ent in ents:
            eid = ent.get("resolved_id", "❓")
            name = ent.get("name", "?")
            typ = ent.get("type", "?")
            role = ent.get("role", "?")
            print(f"   - {name} ({typ}, role={role}) → {eid}")
    else:
        print("🧾 Extracted Entities: none")

    # Constraint Tree
    if getattr(state, 'constraint_tree', None):
        print("📐 Constraint Tree:")
        try:
            tree_str = str(state.constraint_tree)
            if len(tree_str) > 500:
                print(tree_str[:500] + " ... [truncated]")
            else:
                print(tree_str)
        except Exception as e:
            print(f"   [Error printing constraint tree: {e}]")
    else:
        print("📐 Constraint Tree: none")

    # Relaxation Info
    dropped = getattr(state, "last_dropped_constraints", [])
    if dropped:
        print("♻️ Relaxed Constraints:")
        for dc in dropped:
            print(f"   - {dc}")
    else:
        print("♻️ Relaxed Constraints: none")

    # Execution Plan
    steps = getattr(state, "completed_steps", [])
    if steps:
        print("🧭 Completed Steps:")
        for s in steps[:5]:
            print(f"   - {s}")
        if len(steps) > 5:
            print(f"   ... and {len(steps)-5} more")
    else:
        print("🧭 Completed Steps: none")

    # Final Results
    results = getattr(state, "formatted_response", [])
    if results and isinstance(results, list):
        print(f"✅ Final Results: {len(results)} entries")
        for r in results[:3]:
            print(f"   - {str(r)[:100]}...")
        if len(results) > 3:
            print(f"   ... and {len(results)-3} more")
    else:
        print("✅ Final Results: none")

    # Explanation
    explanation = getattr(state, "explanation", None)
    if explanation:
        print(f"🗣️ Explanation: {explanation[:200]}" +
              ("..." if len(explanation) > 200 else ""))
    else:
        print("🗣️ Explanation: none")

    print("=" * 60)
