# from core.model.constraint import evaluate_constraint_tree, relax_constraint_tree
from core.model.constraint import Constraint, ConstraintGroup, ConstraintBuilder
from core.model.evaluator_enhanced import evaluate_constraint_tree, relax_constraint_tree_enhanced


def evaluate_and_inject_from_constraint_tree(state) -> bool:
    """
    Evaluate symbolic constraints against data_registry and inject validation steps.
    Applies constraint relaxation if needed. Returns True if steps were injected.
    """

    # Phase 1: Try constraint tree evaluation directly
    ids_by_key = evaluate_constraint_tree(
        state.constraint_tree, state.data_registry)

    if ids_by_key:
        inject_validation_steps_from_ids(ids_by_key, state)
        for key in ids_by_key:
            for subkey, matches in ids_by_key[key].items():
                if matches:
                    step_type = key.replace("with_", "").rstrip("s")
                    for match_id in matches:
                        state.plan_steps.append({
                            "step_id": f"step_validate_{step_type}_{match_id}",
                            "endpoint": f"/{step_type}/{match_id}",
                            "type": "validation",
                            "produces": ["summary"],
                            "from_constraint_tree": True
                        })
        return True

    # Phase 2: Relax the constraint tree if no matches (Enhanced with multiple drops)
    relaxed_tree, dropped_constraints, reasons = relax_constraint_tree_enhanced(
        state.constraint_tree, max_drops=2, data_registry=state.data_registry)

    if not relaxed_tree:
        return False

    # ✅ Assign relaxed state to planning state
    state.constraint_tree = relaxed_tree
    state.last_dropped_constraints = dropped_constraints

    state.relaxation_log.extend(reasons)

    # Phase 3: Re-run evaluation with relaxed tree
    ids_by_key = evaluate_constraint_tree(
        state.constraint_tree, state.data_registry)

    if ids_by_key:
        inject_validation_steps_from_ids(ids_by_key, state)
        for key in ids_by_key:
            for subkey, matches in ids_by_key[key].items():
                if matches:
                    step_type = key.replace("with_", "").rstrip("s")
                    for match_id in matches:
                        state.plan_steps.append({
                            "step_id": f"step_validate_{step_type}_{match_id}",
                            "endpoint": f"/{step_type}/{match_id}",
                            "type": "validation",
                            "produces": ["summary"],
                            "from_constraint_tree": True
                        })
        return True

    return False


def inject_validation_steps_from_ids(ids_by_key: dict, state) -> None:
    """
    Injects validation steps (e.g., /movie/{id}) based on IDs grouped by TMDB param keys.
    Appends to state.plan_steps.
    """
    for key, id_set in ids_by_key.items():
        if key.startswith("with_movies"):
            media_type = "movie"
        elif key.startswith("with_tv"):
            media_type = "tv"
        else:
            continue  # Skip unknown groups

        for id_ in sorted(id_set):
            step_id = f"step_validate_{media_type}_{id_}"
            if step_id in state.completed_steps:
                continue

            step = {
                "step_id": step_id,
                "endpoint": f"/{media_type}/{id_}",
                "parameters": {},
                "type": "validation",
                "produces": ["summary"],
                "from_constraint_tree": True
            }
            state.plan_steps.insert(0, step)


def intersect_media_ids_across_constraints(results: list, expected: dict, media_type: str) -> list:
    """
    Intersect media results (movies or TV) based on expected constraints:
    - person_ids must match cast/crew
    - company_ids must match production_companies
    - network_ids must match networks (TV only)
    """
    filtered = []

    for result in results:
        match = True

        if "person_ids" in expected:
            cast = result.get("cast", [])
            crew = result.get("crew", [])

            # Extract actual IDs
            cast_ids = {m.get("id") for m in cast if m.get("id")}
            director_ids = {
                m.get("id") for m in crew if m.get("job") == "Director" and m.get("id")
            }

            # Pull from preprocessed structure
            person_by_role = expected.get("person_by_role", {})
            expected_cast_ids = set(person_by_role.get("cast", []))
            expected_director_ids = set(person_by_role.get("director", []))

            # Role-specific matching
            if not expected_cast_ids.issubset(cast_ids):
                match = False
            if not expected_director_ids.issubset(director_ids):
                match = False

        # Company match
        if match and "company_ids" in expected:
            company_ids = {c.get("id") for c in result.get(
                "production_companies", []) if c.get("id")}
            if not any(cid in company_ids for cid in expected["company_ids"]):
                match = False

        # Network match
        if match and media_type == "tv" and "network_ids" in expected:
            network_ids = {n.get("id") for n in result.get(
                "networks", []) if n.get("id")}
            if not any(nid in network_ids for nid in expected["network_ids"]):
                match = False

        if match:
            filtered.append(result)

    return filtered
