# fallback_handler.py
from typing import Dict, List
from copy import deepcopy
from datetime import datetime

class FallbackSemanticBuilder:
    @staticmethod
    def enrich_fallback_step(original_step, extraction_result, resolved_entities):
        """
        Create a smarter fallback discovery step, injecting genres, year, companies/networks if possible.
        """
        intents = extraction_result.get("intents", [])
        query_entities = extraction_result.get("query_entities", [])
        # phase 19.9 - Media Type Enforcement Baseline
        intended_type = extraction_result.get("media_type", "movie")
        if intended_type == "tv":
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

        # Inject genres
        genre_ids = [
            str(e.get("resolved_id"))
            for e in query_entities
            if e.get("type") == "genre" and e.get("resolved_id")
        ]
        if genre_ids:
            fallback_step["parameters"]["with_genres"] = ",".join(genre_ids)

        # Inject year
        date_entities = [
            e for e in query_entities
            if e.get("type") == "date" and e.get("name")
        ]
        if date_entities:
            fallback_step["parameters"][year_param] = date_entities[0]["name"]
        else:
            # 🔥 Phase 19 improvement: default to 5 years ago if no year extracted
            current_year = datetime.now().year
            fallback_step["parameters"][year_param] = str(current_year - 5)

        # Inject company or network
        if resolved_entities.get("company_id"):
            fallback_step["parameters"]["with_companies"] = ",".join(
                str(cid) for cid in resolved_entities["company_id"]
            )
        elif resolved_entities.get("network_id"):
            fallback_step["parameters"]["with_networks"] = ",".join(
                str(nid) for nid in resolved_entities["network_id"]
            )

        # Final debug
        print(f"✨ Smart fallback created: {fallback_step['endpoint']} with params {fallback_step['parameters']}")

        return fallback_step

class FallbackHandler:
    @staticmethod
    def generate_steps(entities: Dict, intents: Dict) -> List[Dict]:
        """Create fallback steps based on available entities."""
        steps = []

        # Entity priority: person > movie > genre > general discover
        if 'person_id' in entities:
            person_ids = entities['person_id']
            if isinstance(person_ids, list) and person_ids:
                fallback_person_id = str(person_ids[0])  # Use only the first person
            elif isinstance(person_ids, int):
                fallback_person_id = str(person_ids)
            else:
                fallback_person_id = None

            if fallback_person_id:
                steps.append({
                    "step_id": f"fallback_discover_movie_{fallback_person_id}",
                    "endpoint": "/discover/movie",
                    "method": "GET",
                    "parameters": {
                        "with_people": fallback_person_id
                    },
                    "fallback_injected": True
                })
                return steps

        elif 'movie_id' in entities:
            movie_ids = entities['movie_id']
            if isinstance(movie_ids, list) and movie_ids:
                fallback_movie_id = str(movie_ids[0])
            elif isinstance(movie_ids, int):
                fallback_movie_id = str(movie_ids)
            else:
                fallback_movie_id = None

            if fallback_movie_id:
                steps.append({
                    "step_id": f"fallback_movie_{fallback_movie_id}",
                    "endpoint": f"/movie/{fallback_movie_id}",
                    "method": "GET",
                    "fallback_injected": True
                })
                return steps

        # Default fallback
        steps.append({
            "step_id": "fallback_discover_general",
            "endpoint": "/discover/movie",
            "method": "GET",
            "parameters": {
                "sort_by": "popularity.desc",
                "page": 1
            },
            "fallback_injected": True
        })

        return steps

    @staticmethod
    def relax_constraints(original_step, already_dropped=None, state=None):
        """
        Given a step that failed validation, return a list of relaxed step(s).
        Now tracks relaxed parameters properly into state.relaxed_parameters.
        """
        from copy import deepcopy

        already_dropped = already_dropped or set()

        relaxation_priority = [
            ("with_companies", "company"),
            ("with_networks", "network"),
            ("with_genres", "genre"),
            ("primary_release_year", "year"),
            ("with_people", "person"),  # LAST TO DROP
        ]

        relaxed_steps = []

        for param_key, label in relaxation_priority:
            if param_key in original_step.get("parameters", {}) and label not in already_dropped:
                relaxed_step = deepcopy(original_step)
                relaxed_step["parameters"].pop(param_key, None)
                relaxed_step["step_id"] = f"{original_step['step_id']}_relaxed_{label}"
                relaxed_steps.append(relaxed_step)
                print(f"♻️ Relaxing {label}: dropped {param_key}")

                # ➡️ NEW: Track relaxation
                if state is not None:
                    if not hasattr(state, "relaxed_parameters"):
                        state.relaxed_parameters = []
                    state.relaxed_parameters.append(label)
                    print(f"📝 Relaxation tracked: {label}")

                    # ➡️ Log in execution trace
                    from execution_orchestrator import ExecutionTraceLogger
                    ExecutionTraceLogger.log_step(
                        relaxed_step["step_id"],
                        path=relaxed_step["endpoint"],
                        status=f"Relaxation Injected (Dropped {param_key})",
                        summary=f"Dropped {param_key} constraint",
                        state=state
                    )

                break  # Only relax one constraint at a time per retry

        return relaxed_steps
    
    @staticmethod
    def enrich_fallback_step(original_step, extraction_result, resolved_entities):
        """
        Create a smart fallback step with semantic enrichment: genres, years, companies/networks.
        """
        intents = extraction_result.get("intents", [])

        # Decide if TV or Movie fallback
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

        query_entities = extraction_result.get("query_entities", []) or []

        # Inject genres if available
        genre_ids = [
            str(e.get("resolved_id"))
            for e in query_entities
            if e.get("type") == "genre" and e.get("resolved_id")
        ]
        if genre_ids:
            fallback_step["parameters"]["with_genres"] = ",".join(genre_ids)

        # Inject year if available
        date_entities = [e for e in query_entities if e.get("type") == "date" and e.get("name")]
        if date_entities:
            fallback_step["parameters"][year_param] = date_entities[0]["name"]

        # Inject company or network if available
        if resolved_entities.get("company_id"):
            fallback_step["parameters"]["with_companies"] = ",".join(str(cid) for cid in resolved_entities["company_id"])
        elif resolved_entities.get("network_id"):
            fallback_step["parameters"]["with_networks"] = ",".join(str(nid) for nid in resolved_entities["network_id"])

        # Safety warning if no enrichment at all
        if not fallback_step["parameters"]:
            print(f"⚠️ Warning: Fallback step for {fallback_step['endpoint']} has no enrichment injected!")

        return fallback_step