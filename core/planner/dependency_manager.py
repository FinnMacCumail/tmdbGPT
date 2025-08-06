# dependency_manager.py
from typing import Any, List
from collections import defaultdict
from core.execution.fallback import FallbackHandler
from core.execution.trace_logger import ExecutionTraceLogger

from core.execution.fallback import relax_roles_and_retry_intersection
from core.planner.constraint_planner import intersect_media_ids_across_constraints
from core.planner.plan_validator import contains_person_role_constraints


def inject_lookup_steps_from_role_intersection(state):
    """
    Phase 21.3: Eager ID intersection for role-based TV/movie lookups.
    - Intersects symbolic results across person, company, and network constraints.
    - Injects /movie/{id} or /tv/{id} lookup steps based on matched entries.
    - Falls back to discovery if no intersection found, even after relaxation.
    """
    # Debug output removed
    # print(f"🧪 Data registry: {state.data_registry}")
    # print(f"🧪 Response IDs: {[r['id'] for r in state.responses]}")
    # print(f"🧪 Intended type: {state.intended_media_type}")

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
                resolved_entities=state.resolved_entities,
                intents=state.extraction_result,
                extraction_result=state.extraction_result
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
        # Debug output removed

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
            media_obj = next(
                (r for r in state.responses if r.get("id") == media_id), {})
            if "first_air_date" in media_obj or media_obj.get("media_type") == "tv":
                validation_steps.append({
                    "step_id": f"step_validate_tv_{media_id}",
                    "endpoint": f"/tv/{media_id}/credits",
                    "produces": ["cast", "crew"],
                    "requires": ["tv_id"]
                })
            else:
                validation_steps.append({
                    "step_id": f"step_validate_movie_{media_id}",
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

        Expand the current execution plan by injecting dependency-based steps
        based on newly resolved entity IDs and symbolic query constraints.

        This method supports two major planning strategies:

        1. 🧩 Role-Aware Credit Step Injection (Phase 20):
        - If the constraint tree contains person-role constraints (e.g., cast, director),
            inject appropriate /person/{id}/{media_type}_credits steps.
        - Each step is tagged with its role and skipped if already satisfied in state.
        - Media type (tv/movie) is respected when constructing the endpoint.

        2. 🧩 Fallback Discovery Injection (Phase 21.1):
        - If company_id or network_id are resolved, inject a /discover/{media_type}
            step with with_companies or with_networks filters accordingly.
        - Used as a symbolic fallback when role-based matching fails.

        Parameters:
            state (AppState): The current app state with resolved entities and constraint tree.
            newly_resolved (dict): Mapping of resolved entity types to IDs (e.g., person_id, company_id).

        Returns:
            List[dict]: A list of injected execution steps (e.g., credits lookups, discovery queries).
        """
        new_steps = []
        query_entities = state.extraction_result.get("query_entities", [])
        media_type = getattr(state, "intended_media_type", "movie")
        constraint_tree = getattr(state, "constraint_tree", None)

        # ✅ Detect role constraints to determine if credit steps should be injected
        inject_roles = contains_person_role_constraints(constraint_tree)

        if inject_roles:
            for key, ids in newly_resolved.items():
                ids = ids if isinstance(ids, list) else [ids]

                for _id in ids:
                    if key == "person_id":
                        # Inject one credit step per role per person_id per media_type
                        roles = {
                            ent.get("role", "actor").lower()
                            for ent in query_entities
                            if ent.get("resolved_id") == _id and ent.get("type") == "person"
                        }

                        for role_tag in roles:
                            if role_tag in getattr(state, "satisfied_roles", set()):
                                # Debug output removed
                                continue

                            target_types = {
                                "movie", "tv"} if media_type == "both" else {media_type}
                            for mtype in target_types:
                                step_id = f"step_{role_tag}_{_id}_{mtype}"
                                endpoint = f"/person/{_id}/{mtype}_credits"
                                produces = [
                                    "movie_id"] if mtype == "movie" else ["tv_id"]

                                new_steps.append({
                                    "step_id": step_id,
                                    "endpoint": endpoint,
                                    "produces": produces,
                                    "requires": ["person_id"],
                                    "role": role_tag,
                                    "media_type": mtype
                                })

                        # Debug output removed
                        if role_tag in getattr(state, "satisfied_roles", set()):
                            # Debug output removed
                            continue

                        # 🔁 Determine target media types
                        target_types = {
                            "movie", "tv"} if media_type == "both" else {media_type}
                        for mtype in target_types:
                            step_id = f"step_{role_tag}_{_id}_{mtype}"
                            endpoint = f"/person/{_id}/{mtype}_credits"
                            produces = [
                                "movie_id"] if mtype == "movie" else ["tv_id"]

                            new_steps.append({
                                "step_id": step_id,
                                "endpoint": endpoint,
                                "produces": produces,
                                "requires": ["person_id"],
                                "role": role_tag,
                                "media_type": mtype
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

        # ✅ Phase 21.1: Fallback discover step based on company/network
        # (still single-valued media_type for fallback)
        fallback_media = "movie" if media_type not in {
            "movie", "tv"} else media_type
        parameters = {}

        if "company_id" in newly_resolved:
            company_ids = newly_resolved["company_id"]
            parameters["with_companies"] = (
                ",".join(map(str, company_ids)) if isinstance(
                    company_ids, list) else str(company_ids)
            )

        if fallback_media == "tv" and "network_id" in newly_resolved:
            network_ids = newly_resolved["network_id"]
            parameters["with_networks"] = (
                ",".join(map(str, network_ids)) if isinstance(
                    network_ids, list) else str(network_ids)
            )

        if parameters:
            discover_step = {
                "step_id": f"step_discover_{fallback_media}_joined",
                "endpoint": f"/discover/{fallback_media}",
                "method": "GET",
                "parameters": parameters,
                "requires": list(parameters.keys()),
                "produces": ["movie_id"] if fallback_media == "movie" else ["tv_id"],
                "from_constraint": "company_network"
            }
            new_steps.append(discover_step)

        # ✅ Inject direct /movie/{id} or /tv/{id} detail lookup
        for ent in query_entities:
            if ent.get("type") == "movie" and ent.get("resolved_id"):
                new_steps.append({
                    "step_id": f"step_movie_details_{ent['resolved_id']}",
                    "endpoint": f"/movie/{ent['resolved_id']}?append_to_response=credits",
                    "produces": ["movie_summary"],
                    "requires": ["movie_id"],
                    "from_constraint": "direct_lookup"
                })
            elif ent.get("type") == "tv" and ent.get("resolved_id"):
                new_steps.append({
                    "step_id": f"step_tv_details_{ent['resolved_id']}",
                    "endpoint": f"/tv/{ent['resolved_id']}",
                    "produces": ["tv_summary"],
                    "requires": ["tv_id"],
                    "from_constraint": "direct_lookup"
                })

        return new_steps
