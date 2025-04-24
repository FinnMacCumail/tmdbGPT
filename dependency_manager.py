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

    def expand_plan_with_dependencies(state, resolved_entities):
        """Plan symbolic joins from role-tagged person entities."""
        query_entities = state.extraction_result.get("query_entities", [])
        plan_steps = state.plan_steps
        new_steps = []

        seen_ids = set()
        for ent in query_entities:
            if ent.get("type") != "person":
                continue
            person_id = ent.get("resolved_id")
            role = ent.get("role") or "cast"
            if person_id and person_id not in seen_ids:
                seen_ids.add(person_id)
                step_id = f"step_{role}_{person_id}"
                new_steps.append({
                    "step_id": step_id,
                    "endpoint": f"/person/{person_id}/movie_credits",
                    "produces": ["movie_id"],
                    "role": role,
                    "from_person_id": person_id,
                    "filters": [{"role": role}]
                })

        # Insert new dependency steps at the start of the plan (or wherever appropriate)
        state.plan_steps = new_steps + plan_steps
        return state