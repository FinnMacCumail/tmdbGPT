def log_summary(state, header=None):
    import logging
    from pprint import pformat

    # Setup logger with fallback StreamHandler
    logger = logging.getLogger("tmdbgpt-summary")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    log = logger.info  # Use this instead of print

    if header:
        log(header)

    log("🧠 DEBUGGING SUMMARY REPORT")
    log("=" * 70)

    # Input
    query = getattr(state, 'query', 'N/A')
    log(f"🔹 Original Query: {query}")

    # Media type
    media_type = getattr(state, 'intended_media_type', 'N/A')
    log(f"🎥 Intended Media Type: {media_type}")

    # Extracted Entities
    ents = getattr(state, "extraction_result", {}).get("query_entities", [])
    if ents:
        log("🧾 Extracted Entities:")
        for ent in ents:
            log(f"   - {ent.get('name', '?')} ({ent.get('type', '?')}, role={ent.get('role', '?')}) → {ent.get('resolved_id', '❓')}")
    else:
        log("🧾 Extracted Entities: none")

    # Constraint Tree
    tree = getattr(state, 'constraint_tree', None)
    if tree:
        log("📐 Constraint Tree:")
        tree_str = str(tree)
        log(tree_str[:500] + (" ..." if len(tree_str) > 500 else ""))
    else:
        log("📐 Constraint Tree: none")

    # Relaxation Info
    dropped = getattr(state, "last_dropped_constraints", [])
    if dropped:
        log("♻️ Relaxed Constraints:")
        for dc in dropped:
            log(f"   - {dc}")
    else:
        log("♻️ Relaxed Constraints: none")

    # Post-validation log
    validations = getattr(state, "post_validation_log", [])
    if validations:
        log("🔬 Post Validations:")
        for val in validations:
            log(f"   - {val}")
    else:
        log("🔬 Post Validations: none")

    # Completed Steps
    steps = getattr(state, "completed_steps", [])
    if steps:
        log(f"🧭 Completed Steps ({len(steps)}):")
        for s in steps[:10]:
            log(f"   - {s}")
        if len(steps) > 10:
            log(f"   ... and {len(steps) - 10} more")
    else:
        log("🧭 Completed Steps: none")

    # Fallback indicators
    used_fallback = any(s.get("fallback_injected") for s in getattr(
        state, "plan_steps", []) if isinstance(s, dict))
    log(f"🛡️ Fallback Injected: {'✅ Yes' if used_fallback else '❌ No'}")

    # Final Results
    results = getattr(state, "formatted_response", [])
    if results and isinstance(results, list):
        log(f"✅ Final Results: {len(results)} entries")
        for r in results[:3]:
            log(f"   - {str(r)[:100]}...")
        if len(results) > 3:
            log(f"   ... and {len(results) - 3} more")
    else:
        log("✅ Final Results: none")

    # Explanation
    explanation = getattr(state, "explanation", None)
    if explanation:
        log(
            f"🗣️ Explanation: {explanation[:200]}{'...' if len(explanation) > 200 else ''}")
    else:
        log("🗣️ Explanation: none")

    # Role Matching Debug
    expected_roles = {
        (qe["role"], qe["resolved_id"])
        for qe in ents
        if qe.get("role") and qe.get("resolved_id")
    }
    log("🎭 Expected Roles: " + (str(expected_roles) if expected_roles else "none"))

    if hasattr(state, "data_registry"):
        for step_id, data in state.data_registry.items():
            if isinstance(data, dict) and ("cast" in data or "crew" in data):
                cast_ids = {p.get("id")
                            for p in data.get("cast", []) if p.get("id")}
                director_ids = {p.get("id") for p in data.get(
                    "crew", []) if p.get("job") == "Director"}
                log(f"   📌 Step {step_id}: Cast={sorted(cast_ids)}, Director={sorted(director_ids)}")

    log("=" * 70)
