# dependency_manager.py
from typing import Any, List
from collections import defaultdict
from core.execution.fallback import FallbackHandler
from core.execution.trace_logger import ExecutionTraceLogger

from core.execution.fallback import relax_roles_and_retry_intersection
from core.planner.constraint_planner import intersect_media_ids_across_constraints


def inject_lookup_steps_from_role_intersection(state):
    """
    Phase 21.3: Eager ID intersection for role-based TV/movie lookups.
    - Intersects symbolic results across person, company, and network constraints.
    - Injects /movie/{id} or /tv/{id} lookup steps based on matched entries.
    - Falls back to discovery if no intersection found, even after relaxation.
    """
    print(f"🧪 Constraint tree: {state.constraint_tree}")
    print(f"🧪 Data registry: {state.data_registry}")
    print(f"🧪 Response IDs: {[r['id'] for r in state.responses]}")
    print(f"🧪 Intended type: {state.intended_media_type}")

    intended_type = getattr(state, "intended_media_type", "both") or "both"
    expected = {
        "person_ids": [],
        "company_ids": [],
        "network_ids": []
    }

    for ent in state.extraction_result.get("query_entities", []):
        if ent.get("type") == "person" and "resolved_id" in ent:
            expected["person_ids"].append(ent["resolved_id"])
        elif ent.get("type") == "company" and "resolved_id" in ent:
            expected["company_ids"].append(ent["resolved_id"])
        elif ent.get("type") == "network" and "resolved_id" in ent:
            expected["network_ids"].append(ent["resolved_id"])

    # 🧩 Primary intersection
    intersection = intersect_media_ids_across_constraints(
        state.responses, expected, intended_type
    )

    # 🔁 Try relaxed roles if no match
    if not intersection:
        relaxed_state = relax_roles_and_retry_intersection(state)
        intersection = intersect_media_ids_across_constraints(
            relaxed_state.responses, expected, intended_type
        )

        if not intersection:
            fallback_steps = FallbackHandler.generate_steps(
                state.resolved_entities,
                intents=state.extraction_result
            )
            if isinstance(fallback_steps, dict):
                fallback_steps = [fallback_steps]

            for fs in reversed(fallback_steps):
                state.plan_steps.insert(0, fs)

            ExecutionTraceLogger.log_step(
                step_id="fallback_injected_after_role_relaxation",
                path="(internal)",
                status="Fallback Injected",
                summary="No matches after relaxing roles. Fallback discovery triggered.",
                state=state
            )
            return state

    # ✅ Inject /tv/{id} or /movie/{id} lookup steps
    injected_ids = []

    for item in intersection:
        item_id = item.get("id")
        if not item_id:
            continue

        is_tv = "first_air_date" in item or "name" in item
        is_movie = "release_date" in item or "title" in item

        if intended_type == "tv" or (intended_type == "both" and is_tv):
            step = {
                "step_id": f"step_lookup_tv_{item_id}",
                "endpoint": f"/tv/{item_id}",
                "method": "GET",
                "produces": [],
                "requires": ["tv_id"]
            }
            state.plan_steps.insert(0, step)
            injected_ids.append(item_id)

        elif intended_type == "movie" or (intended_type == "both" and is_movie):
            step = {
                "step_id": f"step_lookup_movie_{item_id}",
                "endpoint": f"/movie/{item_id}",
                "method": "GET",
                "produces": [],
                "requires": ["movie_id"]
            }
            state.plan_steps.insert(0, step)
            injected_ids.append(item_id)

    # 🧠 Trace successful injection
    if injected_ids:
        ExecutionTraceLogger.log_step(
            step_id="role_intersection_success",
            path="(internal)",
            status="TV/Movie Lookup Injected",
            summary=f"Injected lookup steps for IDs: {injected_ids}",
            state=state
        )
        print(f"✅ Injected role-based lookup steps for IDs: {injected_ids}")

    return state


class DependencyManager:
    @staticmethod
    def analyze_dependencies(state):
        """
        Detect and intersect movie_ids across person role steps.
        Injects /movie/{id}/credits validation steps for common movie IDs.
        """
        person_steps = [step for step in state.completed_steps if step.startswith(
            "step_cast_") or step.startswith("step_director_")]
        movie_id_sets = []
        id_to_step_map = {}

        for step_id in person_steps:
            step_result = state.data_registry.get(step_id, {})
            ids = set()
            for item in step_result.get("crew", []) + step_result.get("cast", []):
                if "id" in item:
                    ids.add(item["id"])
            if ids:
                movie_id_sets.append(ids)
                id_to_step_map[step_id] = ids

        if len(movie_id_sets) < 2:
            return state  # Not enough for intersection

        intersected_ids = set.intersection(*movie_id_sets)
        validation_steps = []

        for media_id in sorted(intersected_ids):
            # Detect whether this ID refers to a movie or TV show
            media_obj = next(
                (r for r in state.responses if r.get("id") == media_id),
                {}
            )
            if "first_air_date" in media_obj or media_obj.get("media_type") == "tv":
                validation_steps.append({
                    "step_id": f"step_validate_tv_{media_id}",
                    "endpoint": f"/tv/{media_id}/credits",
                    "produces": ["cast", "crew"],
                    "requires": ["tv_id"]
                })
            else:
                validation_steps.append({
                    "step_id": f"step_validate_{media_id}",
                    "endpoint": f"/movie/{media_id}/credits",
                    "produces": ["cast", "crew"],
                    "requires": ["movie_id"]
                })

        for movie_id in sorted(intersected_ids):
            validation_steps.append({
                "step_id": f"step_validate_{movie_id}",
                "endpoint": f"/movie/{movie_id}/credits"
            })

        # Inject validation steps to plan
        state.plan_steps = validation_steps + state.plan_steps
        return state

    @staticmethod
    def expand_plan_with_dependencies(state, newly_resolved: dict) -> List[dict]:
        """
        Adds dependency-based symbolic steps:
        - Role-aware credits steps (Phase 20)
        - Collection, TV show steps
        - Discovery fallback enrichment with company/network (Phase 21.1)
        """
        new_steps = []
        query_entities = state.extraction_result.get("query_entities", [])

        # Phase 20: Role-aware person credit steps, TV, collection
        for key, ids in newly_resolved.items():
            if not isinstance(ids, list):
                ids = [ids]

            for _id in ids:
                if key == "person_id":
                    role = "actor"  # default
                    for entity in query_entities:
                        if entity.get("resolved_id") == _id and entity.get("type") == "person":
                            role = entity.get("role", "actor")

                    role_tag = role.lower()
                    # 🔍 DEBUG: Check current role and satisfaction state
                    print(
                        f"🔍 Checking role step for person_id={_id}, role={role_tag}")
                    print("   Already satisfied roles:", getattr(
                        state, "satisfied_roles", set()))
                    # can extend for /tv later

                    if role_tag in getattr(state, "satisfied_roles", set()):
                        print(
                            f"🛑 Skipped /person/{_id}/movie_credits because {role_tag} is already satisfied.")
                        continue

                    endpoint = f"/person/{_id}/movie_credits"
                    step_id = f"step_{role_tag}_{_id}"

                    new_steps.append({
                        "step_id": step_id,
                        "endpoint": endpoint,
                        "produces": ["movie_id"],
                        "requires": ["person_id"],
                        "role": role_tag,
                    })

                elif key == "tv_id":
                    new_steps.append({
                        "step_id": f"step_tv_{_id}",
                        "endpoint": f"/tv/{_id}/credits",
                        "produces": ["cast", "crew"],
                        "requires": ["tv_id"]
                    })

                elif key == "collection_id":
                    new_steps.append({
                        "step_id": f"step_collection_{_id}",
                        "endpoint": f"/collection/{_id}",
                        "produces": ["movie_id"],
                        "requires": ["collection_id"]
                    })

        # Phase 21.1: Symbolic discover fallback injection with company/network
        media_type = getattr(state, "intended_media_type", "movie")
        if media_type not in {"movie", "tv"}:
            media_type = "movie"

        parameters = {}

        if "company_id" in newly_resolved:
            company_ids = newly_resolved["company_id"]
            parameters["with_companies"] = ",".join(map(str, company_ids)) if isinstance(
                company_ids, list) else str(company_ids)

        if media_type == "tv" and "network_id" in newly_resolved:
            network_ids = newly_resolved["network_id"]
            parameters["with_networks"] = ",".join(map(str, network_ids)) if isinstance(
                network_ids, list) else str(network_ids)

        if parameters:
            discover_step = {
                "step_id": f"step_discover_{media_type}_joined",
                "endpoint": f"/discover/{media_type}",
                "method": "GET",
                "parameters": parameters,
                "requires": list(parameters.keys()),
                "produces": ["movie_id"] if media_type == "movie" else ["tv_id"],
                "from_constraint": "company_network"
            }
            new_steps.append(discover_step)

        return new_steps
