# dependency_manager.py
from typing import Any
  
class DependencyManager:
    @staticmethod
    def analyze_dependencies(state):
        """
        Detect and intersect movie_ids across person role steps.
        Injects /movie/{id}/credits validation steps for common movie IDs.
        """
        person_steps = [step for step in state.completed_steps if step.startswith("step_cast_") or step.startswith("step_director_")]
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

        for movie_id in sorted(intersected_ids):
            validation_steps.append({
                "step_id": f"step_validate_{movie_id}",
                "endpoint": f"/movie/{movie_id}/credits"
            })

        # Inject validation steps to plan
        state.plan_steps = validation_steps + state.plan_steps
        return state        

    def expand_plan_with_dependencies(state, newly_resolved: dict) -> list:
        new_steps = []
        query_entities = state.extraction_result.get("query_entities", [])

        for key, ids in newly_resolved.items():
            if not isinstance(ids, list):
                ids = [ids]

            for _id in ids:
                # phase 20 - Role-Aware Multi-Entity Planning and Execution
                if key == "person_id":
                    for _id in ids:
                        role = "actor"  # default
                        for entity in query_entities:
                            if entity.get("resolved_id") == _id and entity.get("type") == "person":
                                role = entity.get("role", "actor")

                        role_tag = role.lower()
                        endpoint = f"/person/{_id}/movie_credits"  # 🔁 can be enhanced for TV later
                        step_id = f"step_{role_tag}_{_id}"

                        new_steps.append({
                            "step_id": step_id,
                            "endpoint": endpoint,
                            "produces": ["movie_id"],
                            "requires": ["person_id"],
                            "role": role_tag,
                        })
                if key == "tv_id":
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
                elif key == "company_id":
                    new_steps.append({
                        "step_id": f"step_company_{_id}",
                        "endpoint": f"/company/{_id}/movies",
                        "produces": ["movie_id"],
                        "requires": ["company_id"]
                    })
                elif key == "network_id":
                    new_steps.append({
                        "step_id": f"step_network_{_id}",
                        "endpoint": f"/network/{_id}/tv",
                        "produces": ["tv_id"],
                        "requires": ["network_id"]
                    })
                elif key in {"company_id", "network_id"}:
                    # 🔁 Inject enriched discovery step instead of raw lookup
                    media_type = getattr(state, "intended_media_type", "movie")

                    step_id = f"step_discover_{media_type}_joined"
                    endpoint = f"/discover/{media_type}"
                    parameters = {}

                    if "company_id" in newly_resolved:
                        company_ids = newly_resolved["company_id"]
                        parameters["with_companies"] = ",".join(map(str, company_ids)) if isinstance(company_ids, list) else str(company_ids)

                    if "network_id" in newly_resolved:
                        network_ids = newly_resolved["network_id"]
                        parameters["with_networks"] = ",".join(map(str, network_ids)) if isinstance(network_ids, list) else str(network_ids)

                    discover_step = {
                        "step_id": step_id,
                        "endpoint": endpoint,
                        "method": "GET",
                        "parameters": parameters,
                        "requires": list(parameters.keys()),  # ["with_companies"] or ["with_networks"]
                        "produces": ["movie_id"] if media_type == "movie" else ["tv_id"]
                    }

                    print(f"🎯 Phase 21.1: Injecting discovery step with company/network → {discover_step}")
                    new_steps.append(discover_step)

        return new_steps
