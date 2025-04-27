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
    
    @staticmethod
    def enrich_fallback_step(original_step, extraction_result, resolved_entities):
        """
        Create a smarter fallback step with genre, year, and correct media type (tv/movie).
        """
        intents = extraction_result.get("intents", [])

        if any("tv" in intent.lower() for intent in intents):
            fallback_endpoint = "/discover/tv"
            year_param = "first_air_date_year"
        else:
            fallback_endpoint = "/discover/movie"
            year_param = "primary_release_year"

        fallback_step = {
            "step_id": f"fallback_{fallback_endpoint.strip('/').replace('/', '_')}",
            "endpoint": fallback_endpoint,
            "parameters": {},
            "fallback_injected": True,
        }

        # Inject genres if present
        genre_ids = [
            str(e["resolved_id"])
            for e in extraction_result.get("query_entities", [])
            if e.get("type") == "genre" and e.get("resolved_id")
        ]
        if genre_ids:
            fallback_step["parameters"]["with_genres"] = ",".join(genre_ids)

        # Inject year if present
        for entity in extraction_result.get("query_entities", []):
            if entity.get("type") == "date" and entity.get("name"):
                fallback_step["parameters"][year_param] = entity["name"]

        # Inject companies or networks if present
        if "company_id" in resolved_entities:
            fallback_step["parameters"]["with_companies"] = ",".join(map(str, resolved_entities["company_id"]))
        elif "network_id" in resolved_entities:
            fallback_step["parameters"]["with_networks"] = ",".join(map(str, resolved_entities["network_id"]))

        return fallback_step