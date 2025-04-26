# fallback_handler.py
from typing import Dict, List
from copy import deepcopy

class FallbackHandler:
    @staticmethod
    def generate_steps(entities: Dict, intents: Dict) -> List[Dict]:  # Remove query_type
        """Create fallback steps based on available entities"""
        steps = []
        
        # Entity priority: person > movie > tv > genre
        if 'person_id' in entities:
            steps.append({
                "step_id": "fallback_person",
                "endpoint": f"/person/{entities['person_id']}",
                "method": "GET"
            })
        elif 'movie_id' in entities:
            steps.append({
                "step_id": "fallback_movie",
                "endpoint": f"/movie/{entities['movie_id']}",
                "method": "GET"
            })
        else:
            steps.append({
                "step_id": "fallback_discover",
                "endpoint": "/discover/movie",
                "method": "GET",
                "parameters": {
                    "sort_by": "popularity.desc",
                    "page": 1
                },
                "fallback_injected": True  # ✅ Add this flag
            })
        return steps

    @staticmethod
    def relax_constraints(original_step: dict, already_dropped: set) -> list:
        """
        Given a step, progressively relax constraints by dropping parameters
        based on a predefined priority: network > company > director > cast.
        """
        print(f"♻️ Relaxing constraints for: {original_step['step_id']}")

        relaxation_priority = [
            "with_networks",
            "with_companies",
            "director_id",   # future placeholder, director needs special handling
            "with_people"
        ]

        current_params = original_step.get("parameters", {}).copy()
        relaxed_steps = []

        for param in relaxation_priority:
            if param in current_params and param not in already_dropped:
                new_step = deepcopy(original_step)
                del new_step["parameters"][param]

                base_id = original_step["step_id"].split("_relaxed_")[0]
                new_suffix = "_relaxed_" + "_relaxed_".join(sorted(already_dropped.union({param})))
                new_step["step_id"] = f"{base_id}{new_suffix}"

                relaxed_steps.append(new_step)
                already_dropped.add(param)

        return relaxed_steps