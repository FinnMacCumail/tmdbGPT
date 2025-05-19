from core.execution.post_validator import PostValidator
from core.planner.plan_validator import should_apply_symbolic_filter
from core.execution.trace_logger import ExecutionTraceLogger
from core.planner.plan_utils import filter_valid_movies
from core.model.constraint import Constraint
from core.planner.constraint_planner import inject_validation_steps_from_ids
from core.model.evaluator import evaluate_constraint_tree
from core.planner.entity_reranker import EntityAwareReranker
from core.entity.param_utils import enrich_symbolic_registry
from core.validation.role_validators import validate_roles
from core.execution.discovery_handler import fetch_credits_for_entity


def handle_discover_tv_step(step, step_id, path, json_data, state, depth, seen_step_keys):

    filtered_tv = PostValidator.run_post_validations(step, json_data, state)

    if not filtered_tv:
        ExecutionTraceLogger.log_step(
            step_id, path, "Filtered", "No matching TV results", state=state
        )
        return

    query_entities = state.extraction_result.get("query_entities", [])
    ranked = EntityAwareReranker.boost_by_entity_mentions(
        filtered_tv, query_entities
    )

    if should_apply_symbolic_filter(state, step):
        valid_shows = filter_valid_movies(
            ranked,
            constraint_tree=state.constraint_tree,
            registry=state.data_registry
        )

        rejected = [show for show in ranked if show not in valid_shows]
        for show in rejected:
            show_id = show.get("id")
            show_name = show.get("name") or "<unknown>"
            print(
                f"❌ [REJECTED] {show_name} (ID: {show_id}) — failed symbolic filter")
    else:
        valid_shows = ranked

    matched_keys = set(
        e.key for e in state.constraint_tree if isinstance(e, Constraint)
    )
    matched = [f"{c.key}={c.value}" for c in state.constraint_tree if isinstance(
        c, Constraint) and c.key in matched_keys]
    relaxed = list(state.relaxation_log)
    validated = list(state.post_validation_log)

    if not hasattr(state, "satisfied_roles"):
        state.satisfied_roles = set()

    for show in valid_shows:
        tv_id = show.get("id")
        show["_step"] = step
        if not tv_id:
            continue

        credits = fetch_credits_for_entity(
            show, state.base_url, state.headers)

        # ✅ Add missing role validation before enrichment
        validate_roles(
            credits=credits,
            query_entities=query_entities,
            movie=show,
            state=state
        )

        # ✅ Then symbolically enrich the registry
        enrich_symbolic_registry(
            show,
            state.data_registry,
            credits=credits,
            keywords=None,
            release_info=None,
            watch_providers=None
        )

        show["final_score"] = show.get("final_score", 1.0)
        show["type"] = "tv_summary"
        show["_provenance"] = {
            "matched_constraints": matched,
            "relaxed_constraints": relaxed,
            "post_validations": validated
        }

        enrich_symbolic_registry(
            show,
            state.data_registry,
            credits=credits,
            keywords=None,
            release_info=None,
            watch_providers=None
        )

        satisfied = show["_provenance"].get("satisfied_roles", [])
        state.satisfied_roles.update(satisfied)

        print(
            f"🧠 Appending validated TV show: {show.get('name')} with score {show.get('final_score')}")
        state.responses.append(show)

    state.data_registry[step_id]["validated"] = ranked

    if state.relaxation_log and (dropped := getattr(state, "last_dropped_constraints", [])):
        restored = []
        for c in dropped:
            ids = state.data_registry.get(
                c.key, {}).get(str(c.value), set())
            if ids:
                state.constraint_tree.constraints.append(c)
                restored.append(f"{c.key}={c.value}")
        already_logged = set(state.relaxation_log)
        for restored_id in restored:
            msg = f"Restored: {restored_id}"
            if msg not in already_logged:
                state.relaxation_log.append(msg)

    if not getattr(state, "constraint_tree_evaluated", False):
        ids = evaluate_constraint_tree(
            state.constraint_tree, state.data_registry)
        if ids:
            inject_validation_steps_from_ids(ids, state)
        state.constraint_tree_evaluated = True

    ExecutionTraceLogger.log_step(
        step_id, path, "Validated", summary=ranked[0], state=state
    )
    state.completed_steps.append(step_id)
