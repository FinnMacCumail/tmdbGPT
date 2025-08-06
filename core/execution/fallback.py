# fallback_handler.py
from typing import Dict, List
from copy import deepcopy
from datetime import datetime
from core.execution.trace_logger import ExecutionTraceLogger
from core.planner.constraint_planner import intersect_media_ids_across_constraints


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

        return fallback_step


class FallbackHandler:
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

                # ➡️ NEW: Track relaxation
                if state is not None:
                    if not hasattr(state, "relaxed_parameters"):
                        state.relaxed_parameters = []
                    state.relaxed_parameters.append(label)

                    # ➡️ Log in execution trace

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
    def build_discover_fallback_step(media_type: str, resolved_entities: dict) -> dict:
        """
        Builds a fallback /discover/movie or /discover/tv step with enriched parameters.
        Adds with_people, with_genres, with_companies, with_networks, etc.
        """
        assert media_type in {"movie", "tv"}

        endpoint = f"/discover/{media_type}"
        params = {}

        # Add cast/person fallback support
        people_ids = [e["id"]
                      for e in resolved_entities.get("person", []) if "id" in e]
        if people_ids:
            params["with_people"] = ",".join(map(str, people_ids))

        # Add genre fallback support
        genre_ids = [e["id"]
                     for e in resolved_entities.get("genre", []) if "id" in e]
        if genre_ids:
            params["with_genres"] = ",".join(map(str, genre_ids))

        # ✅ Phase 21.1: Add company support
        company_ids = [e["id"]
                       for e in resolved_entities.get("company", []) if "id" in e]
        if company_ids:
            params["with_companies"] = ",".join(map(str, company_ids))

        # ✅ Phase 21.1: Add network support (TV only)
        if media_type == "tv":
            network_ids = [e["id"]
                           for e in resolved_entities.get("network", []) if "id" in e]
            if network_ids:
                params["with_networks"] = ",".join(map(str, network_ids))

        return {
            "step_id": f"fallback_discover_{media_type}",
            "endpoint": endpoint,
            "parameters": params,
            "type": "fallback",
            "from_constraint": "fallback_handler.build_discover_fallback_step"
        }

    @staticmethod
    def inject_credit_fallback_steps(state, discover_step):
        """
        After a /discover/movie or /tv step completes, inject /movie/{id}/credits
        steps to enable post-validation with cast/crew data.
        """
        movie_results = state.data_registry.get(
            discover_step["step_id"], {}).get("results", [])
        new_steps = []

            state, "satisfied_roles", set()))
        # ✅ Skip if roles already satisfied via discovery
        if getattr(state, "satisfied_roles", set()) >= {"cast", "director"}:
                  state.satisfied_roles)
            return

        for movie in movie_results:
            movie_id = movie.get("id")
            if not movie_id:
                continue

            step = {
                "step_id": f"step_validate_{movie_id}",
                "endpoint": f"/movie/{movie_id}/credits",
                "depends_on": [discover_step["step_id"]],
                "fallback_injected": True,
                "internal": True,
            }
            new_steps.append(step)

        if new_steps:
            discover_step["fallback_injected"] = True
            state.plan_steps.extend(new_steps)
                f"🔁 Injected {len(new_steps)} fallback credit steps after {discover_step['step_id']}")


    @staticmethod
    def generate_steps(resolved_entities, intents=None, extraction_result=None):
        """
        Generate fallback steps based on resolved entities or user intent.
        Includes an early bail-out for fact-style queries already handled by detail endpoints.
        """
        steps = []

        # 🚫 Early Bail-Out: prevent fallback if fact-style query already hit /movie/{id} or /tv/{id}
        if extraction_result:
            question_type = extraction_result.get("question_type")
            handled = extraction_result.get("__handled_endpoints", [])
            if question_type == "fact" and any(
                ep.startswith("/movie/") or ep.startswith("/tv/") for ep in handled
            ):
                    "🚫 Skipping fallback injection — fact-style query handled via detail endpoint.")
                return []

        # 1️⃣ Entity-based fallbacks
        if "person_id" in resolved_entities:
            for pid in resolved_entities["person_id"]:
                steps.append({
                    "step_id": f"step_person_{pid}",
                    "endpoint": f"/person/{pid}",
                    "method": "GET",
                    "produces": ["person_profile"],
                    "requires": ["person_id"],
                    "fallback_injected": True
                })

        if "company_id" in resolved_entities:
            for cid in resolved_entities["company_id"]:
                steps.append({
                    "step_id": f"step_company_{cid}",
                    "endpoint": f"/company/{cid}",
                    "method": "GET",
                    "produces": ["company_profile"],
                    "requires": ["company_id"],
                    "fallback_injected": True
                })

        if "network_id" in resolved_entities:
            for nid in resolved_entities["network_id"]:
                steps.append({
                    "step_id": f"step_network_{nid}",
                    "endpoint": f"/network/{nid}",
                    "method": "GET",
                    "produces": ["network_profile"],
                    "requires": ["network_id"],
                    "fallback_injected": True
                })

        # 1️⃣.5 Movie/TV entity fallbacks (include credits for fact-style queries like director/creator)
        if "movie_id" in resolved_entities:
            for mid in resolved_entities["movie_id"]:
                steps.append({
                    "step_id": f"step_movie_details_{mid}",
                    "endpoint": f"/movie/{mid}?append_to_response=credits",
                    "method": "GET",
                    "produces": ["movie_summary"],
                    "requires": ["movie_id"],
                    "fallback_injected": True
                })

        if "tv_id" in resolved_entities:
            for tid in resolved_entities["tv_id"]:
                steps.append({
                    "step_id": f"step_tv_details_{tid}",
                    "endpoint": f"/tv/{tid}?append_to_response=credits",
                    "method": "GET",
                    "produces": ["tv_summary"],
                    "requires": ["tv_id"],
                    "fallback_injected": True
                })

        # 2️⃣ Intent-based fallbacks (only if no entity-based steps were generated)
        if not steps:
            fallback_intents = intents.get("intent", []) if intents else []

            if any("trending" in i for i in fallback_intents):
                steps.append({
                    "step_id": "step_trending_fallback",
                    "endpoint": "/trending/movie/day",
                    "method": "GET",
                    "produces": ["movie_summary"],
                    "fallback_injected": True
                })

            elif any("popular" in i for i in fallback_intents):
                steps.append({
                    "step_id": "step_popular_movies",
                    "endpoint": "/movie/popular",
                    "method": "GET",
                    "produces": ["movie_summary"],
                    "fallback_injected": True
                })

            elif any("tv" in i for i in fallback_intents):
                steps.append({
                    "step_id": "step_popular_tv",
                    "endpoint": "/tv/popular",
                    "method": "GET",
                    "produces": ["tv_summary"],
                    "fallback_injected": True
                })

            else:
                # Generic last-ditch fallback
                steps.append({
                    "step_id": "step_generic_trending",
                    "endpoint": "/trending/all/day",
                    "method": "GET",
                    "produces": ["media_summary"],
                    "fallback_injected": True
                })

        return steps


def relax_roles_and_retry_intersection(state):
    """
    Phase 21.4: Relax strict crew roles first, then actor roles if necessary.
    Retries symbolic intersection after each relaxation.
    """
    from core.execution.trace_logger import ExecutionTraceLogger
    from collections import defaultdict

    if not hasattr(state, "relaxed_parameters"):
        state.relaxed_parameters = []

    # 1️⃣ Drop strict crew roles
    for role_prefix in ["step_director_", "step_writer_", "step_producer_", "step_composer_"]:
        for step_id in list(state.completed_steps):
            if step_id.startswith(role_prefix):
                state.completed_steps.remove(step_id)
                state.data_registry.pop(step_id, None)

                role_name = role_prefix.replace("step_", "").replace("_", "")
                if role_name not in state.relaxed_parameters:
                    state.relaxed_parameters.append(role_name)

                ExecutionTraceLogger.log_step(
                    step_id=step_id,
                    path="(internal)",
                    status="Role Relaxed",
                    summary=f"Dropped strict crew role: {role_name}",
                    state=state
                )

    # 2️⃣ Retry intersection
    intended_type = getattr(state, "intended_media_type", "movie")

    expected = {
        "person_ids": [
            e["resolved_id"]
            for e in state.extraction_result.get("query_entities", [])
            if e.get("type") == "person"
        ],
        "company_ids": state.resolved_entities.get("company_id", []),
        "network_ids": state.resolved_entities.get("network_id", []),
        "person_by_role": defaultdict(list)
    }

    for q in state.extraction_result.get("query_entities", []):
        if q.get("type") == "person" and q.get("role") and "resolved_id" in q:
            expected["person_by_role"][q["role"]].append(q["resolved_id"])

    intersection = intersect_media_ids_across_constraints(
        state.responses, expected, intended_type
    )

    if intersection:
        return state  # Success after crew-role relaxation

    # 3️⃣ Drop actor (cast) roles if still no match
    for step_id in list(state.completed_steps):
        if step_id.startswith("step_cast_"):
            state.completed_steps.remove(step_id)
            state.data_registry.pop(step_id, None)

            if "cast" not in state.relaxed_parameters:
                state.relaxed_parameters.append("cast")

            ExecutionTraceLogger.log_step(
                step_id=step_id,
                path="(internal)",
                status="Role Relaxed (Cast)",
                summary="Dropped actor role: step_cast",
                state=state
            )

    return state
