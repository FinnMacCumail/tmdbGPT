from typing import TYPE_CHECKING, Dict, Set, List
from collections import defaultdict
from nlp_retriever import PostStepUpdater, PathRewriter, ResultExtractor, expand_plan_with_dependencies
import requests
from copy import deepcopy
from hashlib import sha256
from post_validator import PostValidator
from entity_reranker import EntityAwareReranker
from plan_validator import PlanValidator
import json
from response_formatter import RESPONSE_RENDERERS, format_fallback
from fallback_handler import FallbackHandler, FallbackSemanticBuilder
from post_validator import PostValidator
from response_formatter import QueryExplanationBuilder
from plan_validator import SymbolicConstraintFilter
from constraint_model import evaluate_constraint_tree, relax_constraint_tree
from typing import TYPE_CHECKING
from dependency_manager import DependencyManager
from constraint_model import Constraint, ConstraintGroup
import hashlib
from pydantic import BaseModel
from param_utils import update_symbolic_registry
from log_summary import log_summary


class ExecutionOrchestrator:

    VALIDATION_REGISTRY = [
        {
            "endpoint": "/discover/movie",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/movie/{movie_id}/credits",
            "validator": PostValidator.has_all_cast,
            "args_builder": lambda step, state: {
                "required_ids": [
                    int(p) for p in step["parameters"].get("with_people", "").split(",") if p.isdigit()
                ]
            },
            "arg_source": "credits"
        },
        {
            "endpoint": "/discover/movie",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/movie/{movie_id}/credits",
            "validator": PostValidator.has_director,
            "args_builder": lambda step, state: {
                "director_name": next((
                    e["name"] for e in state.extraction_result.get("query_entities", [])
                    if e.get("type") == "person" and e.get("role") == "director"
                ), None)
            },
            "arg_source": "credits"
        },
        {
            "endpoint": "/discover/tv",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/tv/{tv_id}/credits",
            "validator": PostValidator.has_all_cast,
            "args_builder": lambda step, state: {
                "required_ids": [
                    int(p) for p in step["parameters"].get("with_people", "").split(",") if p.isdigit()
                ]
            },
            "arg_source": "credits"
        }
    ]

    def __init__(self, base_url, headers):
        from dependency_manager import DependencyManager
        self.dependency_manager = DependencyManager()
        self.base_url = base_url
        self.headers = headers
        self.validator = PlanValidator()

    def _run_post_validations(self, step, data, state):
        validated = []
        results = data.get("results", [])
        query_entities = state.extraction_result.get("query_entities", [])

        for rule in self.VALIDATION_REGISTRY:
            if rule["endpoint"] in step["endpoint"] and rule["trigger_param"] in step.get("parameters", {}):
                validator = rule["validator"]
                build_args = rule["args_builder"]
                args = build_args(step, state)

                for item in results:
                    item_id = item.get("id")
                    if not item_id:
                        continue

                    url_template = rule["followup_endpoint_template"]
                    url = f"{self.base_url}{url_template.replace('{tv_id}', str(item_id)).replace('{movie_id}', str(item_id))}"

                    try:
                        response = requests.get(url, headers=self.headers)
                        if response.status_code != 200:
                            continue

                        result_data = response.json()
                        score_tuple = self._score_movie_against_query(
                            movie=item,
                            state=state,
                            credits=result_data,
                            step=step,
                            query_entities=query_entities
                        )

                        if not score_tuple:
                            continue

                        score, matched = score_tuple
                        if score > 0:
                            item["final_score"] = min(score, 1.0)

                            post_validations = item.setdefault(
                                "_provenance", {}).setdefault("post_validations", [])
                            if rule["validator"].__name__ == "has_all_cast":
                                post_validations.append("has_all_cast")
                            elif rule["validator"].__name__ == "has_director":
                                post_validations.append("has_director")

                            validated.append(item)

                    except Exception as e:
                        print(f"⚠️ Validation failed for ID={item_id}: {e}")

                break  # Only apply the first matching rule

        return validated or results

    def execute(self, state):
        # print(log_summary(state, header="🚀 Starting Execution"))
        state.error = None
        state.data_registry = {}
        state.completed_steps = []
        seen_step_keys = set()
        step_origin_depth = {}
        MAX_CHAIN_DEPTH = 3

        # print(f"🧭 Question Type: {getattr(state, 'question_type', None)}")
        # print(f"🎨 Response Format: {getattr(state, 'response_format', None)}")

        # ✅ Phase 21.5: Evaluate and inject constraint-tree-based steps (with relaxation fallback)
        # if self._evaluate_and_inject_from_constraint_tree(state):
        #     print("✅ Constraint tree steps injected.")
        # else:
        #     print("🛑 No executable steps from constraint tree.")

        # ✅ phase 9.2 - pgpv - Safety check happens AFTER constraint planning
        # if not self._safe_to_execute(state):
        #     # print(f"🛑 Fallback triggered due to unsafe plan.")
        #     return state

        print("🧪 About to execute steps:", [
              s["step_id"] for s in state.plan_steps])
        print("🎯 Intended media type:", state.intended_media_type)
        while state.plan_steps:
            step = state.plan_steps.pop(0)  # process from front

            # phase 19.9 - Media Type Enforcement Baseline
            endpoint = step.get("endpoint")
            if not endpoint:
                continue  # Skip steps without an endpoint

            # Skip mismatched media types if enforced
            if state.intended_media_type and state.intended_media_type != "both":
                resolved_path = PathRewriter.rewrite(
                    endpoint, state.resolved_entities) or ""
                print(f"🎯 [Media Filter] Resolved path: {resolved_path}")

                if "/tv" in resolved_path and state.intended_media_type != "tv":
                    print(
                        f"⏭️ Skipping TV step for movie query: {resolved_path}")
                    continue
                if "/movie" in resolved_path and state.intended_media_type != "movie":
                    print(
                        f"⏭️ Skipping movie step for TV query: {resolved_path}")
                    continue

            step_id = step.get("step_id")

            # 🧩 pase 4 pgpv - NEW: Check if required entities are missing
            missing_requires = [
                req for req in step.get("requires", [])
                if req not in state.resolved_entities
            ]

            if missing_requires:
                # 🧠 NEW: Soft Relaxation Phase 10
                soft_filters = {"genre", "date", "runtime",
                                "votes", "rating", "language", "country"}
                soft_missing = []

                for req in missing_requires:
                    entity_type = SymbolicConstraintFilter._map_key_to_entity(
                        req)
                    if entity_type in soft_filters:
                        soft_missing.append(req)

                if soft_missing and len(soft_missing) == len(missing_requires):
                    # print(f"⚡ Soft relaxation: missing only soft filters {soft_missing}. Proceeding with relaxed step.")
                    # ✅ Mark the step as relaxed so post-filtering can occur later
                    step.setdefault("soft_relaxed", []).extend(soft_missing)
                else:
                    # print(f"⏭️ Skipping step {step_id}: missing required core entities {missing_requires}")
                    continue  # Skip hard requirements

            if step_id in state.completed_steps:
                # print(f"✅ Skipping already completed step: {step_id}")
                continue
            # Skip param injection if already relaxed
            if "_relaxed" not in step.get("step_id", ""):
                step = self.validator.inject_path_slot_parameters(
                    step,
                    resolved_entities=state.resolved_entities,
                    extraction_result=state.extraction_result
                )

            step_id = step.get("step_id")
            depth = step_origin_depth.get(step_id, 0)
            if depth > MAX_CHAIN_DEPTH:
                continue

            # 🛡 Sanity check on parameters
            params = step.get("parameters", {})
            if not isinstance(params, dict):
                # print(f"🚨 Malformed parameters in step {step_id} → {type(params)}")
                params = {}

            # 🧠 Replace placeholders in the path using updated params
            path = step.get("endpoint")
            for k, v in params.items():
                if f"{{{k}}}" in path:
                    # Fix: handle list injection
                    value = v[0] if isinstance(v, list) else v
                    path = path.replace(f"{{{k}}}", str(value))
                    # print(f"🧩 Replaced path slot: {{{k}}} → {v}")
            # print(f"🛠️ Resolved full path: {path}")
            path = PathRewriter.rewrite(path, state.resolved_entities)
            full_url = f"{self.base_url}{path}"
            # print(f"\n⚡ Executing {step_id}: {path}")

            # Sanitize structured query parameter
            if isinstance(params.get("query"), dict):
                original = params["query"]
                params["query"] = original.get("name", "")
                # print(f"🔧 Flattened structured query param from {original} → '{params['query']}'")

            # ✅ Deduplication AFTER path + param injection
            param_string = "&".join(
                f"{k}={v}" for k, v in sorted(params.items()))
            dedup_key = f"{step['endpoint']}?{param_string}"
            step_hash = sha256(dedup_key.encode()).hexdigest()

            if step_hash in seen_step_keys:
                # print(f"🔁 Skipping duplicate step_id {step_id} (hash={step_hash}) with same parameters")
                continue

            seen_step_keys.add(step_hash)

            try:
                # print(f"📤 Calling TMDB: {full_url}")
                # print(f"📦 Params: {params}")
                response = requests.get(
                    full_url, headers=self.headers, params=params)

                if response.status_code == 200:
                    # print(f"✅ Success: {response.status_code}")
                    try:
                        json_data = response.json()
                        state.data_registry[step_id] = json_data

                        previous_entities = set(state.resolved_entities.keys())
                        state = PostStepUpdater.update(state, step, json_data)
                        new_entities = {
                            k: v for k, v in state.resolved_entities.items()
                            if k not in previous_entities
                        }

                        # 🧠 Handle step-specific logic
                        if step["endpoint"].startswith("/discover/movie"):
                            self._handle_discover_movie_step(
                                step, step_id, path, json_data, state, depth, seen_step_keys)
                        else:
                            self._handle_generic_response(
                                step, step_id, path, json_data, state)
                        # ✅ Step now completed → safe to check for role intersection

                        # 👇 Fallback credit step injection goes here — AFTER main response handling
                        if step["endpoint"].startswith("/discover/movie") and not step.get("fallback_injected"):
                            FallbackHandler.inject_credit_fallback_steps(
                                state, step)
                            step["fallback_injected"] = True

                        if not step.get("fallback_injected"):
                            expected_role_steps = {
                                f"step_{qe['role']}_{qe['resolved_id']}"
                                for qe in state.extraction_result.get("query_entities", [])
                                if qe.get("type") == "person" and qe.get("role")
                            }

                            if expected_role_steps.issubset(set(state.completed_steps)):
                                print(
                                    "✅ All symbolic role steps complete → triggering intersection")
                                state = DependencyManager.analyze_dependencies(
                                    state)
                                state = self._inject_lookup_steps_from_role_intersection(
                                    state)
                        # Append new steps if needed
                        if new_entities:
                            new_steps = expand_plan_with_dependencies(
                                state, new_entities)
                            if new_steps:
                                # print(f"🔁 Appending {len(new_steps)} new dependent step(s) to execution queue.")
                                for new_step in new_steps:
                                    state.plan_steps.append(new_step)
                                    step_origin_depth[new_step["step_id"]
                                                      ] = depth + 1

                    except Exception as ex:
                        print(f"⚠️ Could not parse JSON or update state: {ex}")
            except Exception as ex:
                print(f"🔥 Step {step_id} failed with exception: {ex}")
                ExecutionTraceLogger.log_step(
                    step_id, path, f"Failed ({str(ex)})", state=state
                )
                state.error = str(ex)

        # 👇 Safely determine the format type from state
        # format_type = getattr(state, "response_format", "summary")
        format_type = state.response_format or "summary"
        renderer = RESPONSE_RENDERERS.get(format_type, format_fallback)

        # 👇 Generate final formatted output
        final_output = renderer(state)

        # 👇 You can optionally assign it to state if needed
        state.formatted_response = final_output

        state.explanation = QueryExplanationBuilder.build_final_explanation(
            extraction_result=state.extraction_result,
            relaxed_parameters=state.relaxed_parameters,
            fallback_used=any(step.get("fallback_injected")
                              for step in state.plan_steps)
        )

        # print("[DEBUG] Orchestrator execution completed.")
        # phase 17.4 - logging
        if getattr(state, "relaxed_parameters", []):
            relaxed_summary = ", ".join(sorted(set(state.relaxed_parameters)))
            ExecutionTraceLogger.log_step(
                "relaxation_summary",
                path="(global)",  # not tied to a specific API path
                status="Relaxation Summary",
                summary=f"Relaxed constraints attempted before fallback: {relaxed_summary}",
                state=state
            )

        log_summary(state)
        return state

    def _intersect_media_ids_across_constraints(self, results: list, expected: dict, media_type: str) -> list:
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

    def _score_movie_against_query(self, movie, state, credits=None, **kwargs):
        matched_constraints = []
        relaxed = getattr(state, "last_dropped_constraints", [])

        # Check person role (e.g., director) constraints
        for constraint in state.constraint_tree.flatten():
            if constraint.type == "person" and constraint.subtype == "director":
                if credits:
                    directors = [p for p in credits.get(
                        "crew", []) if p.get("job") == "Director"]
                    if any(str(p["id"]) == str(constraint.value) for p in directors):
                        matched_constraints.append(
                            f"{constraint.key}={constraint.value}")
                    else:
                        return 0, []  # No match
                else:
                    return 0, []  # No credits to check against

        # Check genre constraints
        if "genre_ids" in movie:
            for constraint in state.constraint_tree.flatten():
                if constraint.type == "genre":
                    if int(constraint.value) in movie["genre_ids"]:
                        matched_constraints.append(
                            f"{constraint.key}={constraint.value}")

        for constraint in state.constraint_tree.flatten():
            if constraint.type == "company":
                if not PostValidator.validate_company(movie, [constraint.value]):
                    return 0, []
                matched_constraints.append(
                    f"{constraint.key}={constraint.value}")

            # Network constraint validation (TV only)
            if movie.get("media_type") == "tv":
                for constraint in state.constraint_tree.flatten():
                    if constraint.type == "network":
                        if not PostValidator.validate_network(movie, [constraint.value]):
                            return 0, []
                        matched_constraints.append(
                            f"{constraint.key}={constraint.value}")

        # Track provenance
        movie["_provenance"] = movie.get("_provenance", {})
        movie["_provenance"]["matched_constraints"] = matched_constraints
        movie["_provenance"]["relaxed_constraints"] = [
            f"Dropped '{c.key}={c.value}' (priority={c.priority}, confidence={c.confidence})"
            for c in relaxed
        ]

        # Begin post-validation tracking
        post_validations = []
        if credits:
            # Add role-based validation
            cast_ids = {m.get("id") for m in credits.get("cast", [])}
            crew = credits.get("crew", [])
            director_ids = {p.get("id")
                            for p in crew if p.get("job") == "Director"}

            for constraint in state.constraint_tree.flatten():
                if constraint.type == "person" and constraint.subtype == "actor":
                    if int(constraint.value) in cast_ids:
                        post_validations.append("has_all_cast")
                if constraint.type == "person" and constraint.subtype == "director":
                    if int(constraint.value) in director_ids:
                        post_validations.append("has_director")

        # Add company/network matches
        for constraint in state.constraint_tree.flatten():
            if constraint.type == "company":
                post_validations.append("company_matched")
            if constraint.type == "network":
                post_validations.append("network_matched")

        movie["_provenance"]["post_validations"] = post_validations

        return len(matched_constraints), matched_constraints

    # phase 21.5 - ID injection logic

    def _inject_validation_steps_from_ids(self, ids_by_key, state):
        for key, id_set in ids_by_key.items():
            # Determine the media type based on the key
            if key.startswith("with_movies"):
                media_type = "movie"
            elif key.startswith("with_tv"):
                media_type = "tv"
            else:
                # print(f"⏭️ Skipping non-media validation group: {key}")
                continue

            for id_ in sorted(id_set):
                step_id = f"step_validate_{media_type}_{id_}"
                if step_id in state.completed_steps:
                    continue

                step = {
                    "step_id": step_id,
                    "endpoint": f"/{media_type}/{id_}",
                    "parameters": {},
                    "type": "validation"
                }
                state.plan_steps.insert(0, step)
                # print(f"✅ Injected validation step: {step_id}")

    # phase 21.5 - Constraint-aware fallback / relaxation

    def _evaluate_and_inject_from_constraint_tree(self, state):
        """
        Evaluates the constraint tree against the symbolic registry,
        injects validation steps for matching IDs, and tracks relaxation state.
        """
        # print("🌿 Evaluating constraint tree against symbolic registry...")

        ids_by_key = evaluate_constraint_tree(
            state.constraint_tree, state.data_registry)

        if ids_by_key:
            # print(f"🎯 Constraint evaluation matched symbolic IDs: {ids_by_key}")
            self._inject_validation_steps_from_ids(ids_by_key, state)
            return True

        # print("🛑 No matches found — attempting constraint relaxation...")

        relaxed_tree, dropped_constraints, reasons = relax_constraint_tree(
            state.constraint_tree)

        state.last_dropped_constraints = dropped_constraints

        if not relaxed_tree:
            # print("🚫 Constraint relaxation failed — no constraints could be dropped.")
            return False

        state.constraint_tree = relaxed_tree
        state.relaxation_log.extend(reasons)

        # print(
        #     f"♻️ Relaxation applied. Dropped constraints: {[f'{c.key}={c.value}' for c in dropped_constraints]}")
        # print(f"📜 Relaxation reasons: {reasons}")

        ids_by_key = evaluate_constraint_tree(
            state.constraint_tree, state.data_registry)

        if ids_by_key:
            # print(f"🎯 Post-relaxation match: {ids_by_key}")
            self._inject_validation_steps_from_ids(ids_by_key, state)
            return True

        # print("🛑 Even after relaxation, no symbolic matches found.")
        return False

    # phase Phase 21.5.8: Smart Step Pruning

    def _make_constraint_fingerprint(self, tree: ConstraintGroup) -> str:

        class _Serializable(BaseModel):
            logic: str
            constraints: List[Dict]

        # Convert constraint objects to dicts
        constraints = sorted(
            # Use c.to_dict() instead of c.dict()
            [c.to_dict() for c in tree.constraints],
            key=lambda d: (d["key"], str(d["value"]))
        )
        tree_repr = _Serializable(
            logic=tree.logic, constraints=constraints).json()
        return hashlib.md5(tree_repr.encode()).hexdigest()

    def _handle_discover_movie_step(self, step, step_id, path, json_data, state, depth=0, seen_step_keys=None):
        seen_step_keys = seen_step_keys or set()
        # print(f"🔎 BEGIN _handle_discover_movie_step for {step_id}")

        # Phase 1: Post-validation
        filtered_movies = self._run_post_validations(step, json_data, state)
        if not filtered_movies:
            # print("⚠️ No valid results matched required cast/director.")
            ExecutionTraceLogger.log_step(
                step_id, path, "Filtered", "No matching results", state=state
            )
            state.responses.append(
                "⚠️ No valid results matched all required cast/director.")

            already_dropped = {p.strip()
                               for p in step_id.split("_relaxed_")[1:] if p}
            relaxed_steps = FallbackHandler.relax_constraints(
                step, already_dropped, state=state)

            if relaxed_steps:
                for relaxed_step in relaxed_steps:
                    if relaxed_step["step_id"] not in state.completed_steps:
                        constraint_dropped = relaxed_step["step_id"].split("_relaxed_")[
                            1]
                        # print(f"♻️ Injected relaxed retry: {relaxed_step['step_id']} (Dropped {constraint_dropped})")
                        state.plan_steps.insert(0, relaxed_step)
                        ExecutionTraceLogger.log_step(
                            relaxed_step["step_id"], path,
                            status=f"Relaxation Injected ({constraint_dropped})",
                            summary=f"Dropped constraint: {constraint_dropped}",
                            state=state
                        )
                state.relaxed_parameters.extend(already_dropped)
                ExecutionTraceLogger.log_step(
                    step_id, path, "Relaxation Started", summary="Injected relaxed steps", state=state
                )
                state.completed_steps.append(step_id)
                # print(f"✅ Marked original step completed after injecting relaxed retries.")
                return

            # No more relaxation → fallback
            # print("🛑 All filter drop retries exhausted. Injecting semantic fallback...")
            fallback_step = FallbackSemanticBuilder.enrich_fallback_step(
                original_step=step,
                extraction_result=state.extraction_result,
                resolved_entities=state.resolved_entities
            )
            if fallback_step["step_id"] not in state.completed_steps:
                state.plan_steps.insert(0, fallback_step)
                ExecutionTraceLogger.log_step(
                    fallback_step["step_id"], fallback_step["endpoint"],
                    status="Semantic Fallback Injected",
                    summary=f"Enriched fallback injected with parameters: {fallback_step.get('parameters', {})}",
                    state=state
                )
                # print(f"🧭 Injected enriched fallback step: {fallback_step['endpoint']}")
            # else:
            #     print("⚠️ Fallback already completed — skipping reinjection.")

            state.completed_steps.append(step_id)
            # print(f"✅ Marked as completed: {step_id}")
            return

        # Phase 2: Rank and boost
        # print(f"✅ Found {len(filtered_movies)} filtered result(s)")
        query_entities = state.extraction_result.get("query_entities", [])
        ranked = EntityAwareReranker.boost_by_entity_mentions(
            filtered_movies, query_entities)

        # Phase 3: Symbolic registry + provenance
        ids = evaluate_constraint_tree(
            state.constraint_tree, state.data_registry)
        matched_keys = set(ids.keys())
        matched = []
        for c in state.constraint_tree:
            if isinstance(c, Constraint) and c.key in matched_keys:
                matched.append(f"{c.key}={c.value}")
        relaxed = list(state.relaxation_log)
        validated = list(state.post_validation_log)

        for movie in ranked:
            movie["final_score"] = movie.get("final_score", 1.0)
            movie["type"] = "movie_summary"
            movie["_provenance"] = {
                "matched_constraints": matched,
                "relaxed_constraints": relaxed,
                "post_validations": validated
            }
            update_symbolic_registry(movie, state.data_registry)
            print(
                f"🧠 Appending validated movie: {movie.get('title')} with score {movie.get('final_score')}")
            state.responses.append(movie)

        # Phase 4: Save validated results
        state.data_registry[step_id]["validated"] = ranked

        # ✅ Phase 5: Restore dropped constraints if now satisfied
        if state.relaxation_log and (dropped := getattr(state, "last_dropped_constraints", [])):
            restored = []
            for c in dropped:
                ids = state.data_registry.get(
                    c.key, {}).get(str(c.value), set())
                if ids:
                    state.constraint_tree.constraints.append(c)
                    restored.append(f"{c.key}={c.value}")
            if restored:
                # print(f"🔁 Restored relaxed constraints that now match: {restored}")
                already_logged = set(state.relaxation_log)
                for restored_id in restored:
                    msg = f"Restored: {restored_id}"
                    if msg not in already_logged:
                        state.relaxation_log.append(msg)

        # Phase 6: Re-evaluate constraints (deferred)
        if not getattr(state, "constraint_tree_evaluated", False):
            ids = evaluate_constraint_tree(
                state.constraint_tree, state.data_registry)
            if ids:
                # print(f"🔄 Deferred Constraint Evaluation → matched IDs: {ids}")
                self._inject_validation_steps_from_ids(ids, state)
            # else:
            #     print("🛑 No constraint matches after deferred evaluation.")
            state.constraint_tree_evaluated = True

        # Final log
        ExecutionTraceLogger.log_step(
            step_id, path, "Validated", summary=ranked[0], state=state
        )
        state.completed_steps.append(step_id)
        # print(f"✅ Step marked completed: {step_id}")

    def _handle_generic_response(self, step, step_id, path, json_data, state):
        print(f"🛑 Check path before rewrite →  {path}")
        path = PathRewriter.rewrite(step["endpoint"], state.resolved_entities)
        print(f"🛑 Check path after rewrite →  {path}")
        summaries = []
        filtered_summaries = []

        try:
            print(f"🧪 ResultExtractor.extract called with endpoint: {path}")
            summaries = ResultExtractor.extract(
                path, json_data, state.resolved_entities)
            print(f"🧪 Extracted {len(summaries)} summaries")
        except Exception as e:
            print(f"⚠️ Failed during extract(): {e}")
            print(f"⚠️ Path: {path}")

        try:
            applied_params = state.extraction_result.get(
                "applied_parameters", {})
            should_filter = ResultExtractor.should_post_filter(
                step["endpoint"], applied_params)
            print(f"🧪 Should post-filter? {should_filter}")

            if summaries and should_filter:
                filtered_summaries = ResultExtractor.post_filter_responses(
                    summaries,
                    query_entities=state.extraction_result.get(
                        "query_entities", []),
                    extraction_result=state.extraction_result
                )
                print(
                    f"📊 Post-filtered to {len(filtered_summaries)} summaries")
            else:
                filtered_summaries = summaries
                print(f"📊 Using unfiltered summaries")

            state.responses.extend(filtered_summaries)

        except Exception as e:
            print(f"⚠️ Failed during post-filtering: {e}")

            state.responses.extend(filtered_summaries)

        except Exception as e:
            print(f"⚠️ Could not parse JSON or update state: {e}")

        # 🧪 Optional: Run post-validation for /discover/tv with cast
        if step["endpoint"].startswith("/discover/tv") and "with_people" in step.get("parameters", {}):
            try:
                # print(f"🧪 Running TV post-validation for {step['endpoint']}")
                validated_summaries = self._run_post_validations(
                    step, {"results": filtered_summaries}, state
                )

                if not validated_summaries:
                    # print("🛑 No validated results after cast-check — injecting fallback...")
                    fallback_step = FallbackSemanticBuilder.enrich_fallback_step(
                        original_step=step,
                        extraction_result=state.extraction_result,
                        resolved_entities=state.resolved_entities
                    )

                    if fallback_step["step_id"] not in state.completed_steps:
                        state.plan_steps.insert(0, fallback_step)

                    state.completed_steps.append(step_id)
                    return

                state.responses.extend(validated_summaries)
                # ✅ Symbolic registry tracking for validated TV shows
                if step["endpoint"].startswith("/discover/tv"):
                    tv_ids = [item.get("id")
                              for item in validated_summaries if item.get("id")]
                    if tv_ids:
                        # print(f"📦 Tracking TV IDs: {tv_ids}")
                        registry = state.data_registry.setdefault(
                            "tv_ids", set())
                        registry.update(tv_ids)
            except Exception as e:
                # print(f"❌ ERROR during post-validation: {e}")
                state.responses.extend(filtered_summaries)
        else:
            state.responses.extend(filtered_summaries)

        # ✅ NEW: Handle post-validation producers (e.g., /movie/{id}/credits)
        validated = state.data_registry.get(step_id, {}).get("validated")
        if validated:
            print(
                f"🧠 Appending {len(validated)} validated fallback movie(s) from {step_id}")
            query_entities = state.extraction_result.get("query_entities", [])
            reranked = EntityAwareReranker.boost_by_entity_mentions(
                validated, query_entities)

            # 🧹 Deduplicate by TMDB movie ID
            seen_ids = {resp["id"] for resp in state.responses if "id" in resp}
            reranked = [movie for movie in reranked if movie.get(
                "id") not in seen_ids]

            for movie in reranked:
                movie["final_score"] = movie.get("final_score", 1.0)
                movie["type"] = "movie_summary"
                movie["_provenance"] = {
                    "source": "post_validation",
                    "via": step_id
                }
                state.responses.append(movie)

        ExecutionTraceLogger.log_step(
            step_id, path, "Handled", filtered_summaries[:1], state=state
        )
        state.completed_steps.append(step_id)

    def _inject_validation_steps(self, state, intersected_ids: set) -> None:
        """
        After intersecting movie/tv IDs, inject validation steps for the survivors.
        """
        validation_steps = []

        for idx, media_id in enumerate(sorted(intersected_ids)):
            validation_steps.append({
                "step_id": f"step_validate_{media_id}",
                # 🛠 expand to TV later too
                "endpoint": f"/movie/{media_id}/credits",
                "method": "GET",
                "produces": ["cast", "crew"],
                "requires": ["movie_id"],  # expandable later
                "from_intersection": True  # ✅ Tag it
            })

        # Insert validation steps at the beginning of plan queue
        # print(f"✅ Injecting {len(validation_steps)} validation step(s) after intersection.")
        state.plan_steps = validation_steps + state.plan_steps

    # def _safe_to_execute(self, state):
    #     results = state.data_registry.get(
    #         "step_discover_movie", {}).get("results", [])
    #     media_type = getattr(state, "intended_media_type", "movie")
    #     expected = {
    #         "person_ids": [e["resolved_id"] for e in state.extraction_result.get("query_entities", []) if e["type"] == "person"],
    #         "company_ids": state.resolved_entities.get("company_id", []),
    #         "network_ids": state.resolved_entities.get("network_id", [])
    #     }
    #     filtered = self._intersect_media_ids_across_constraints(
    #         results, expected, media_type)
    #     return bool(filtered)

    def _inject_lookup_steps_from_role_intersection(self, state):
        """
        Refactored for Phase 21.3:
        - Intersects symbolic results across person, company, network
        - Injects /movie/{id} or /tv/{id} lookup steps based on matched entries
        - Applies fallback if no match after relaxing constraints
        """
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

        qe = state.extraction_result.get("query_entities", [])
        expected["person_by_role"] = defaultdict(list)

        for q in qe:
            if q.get("type") == "person" and q.get("resolved_id") and q.get("role"):
                expected["person_by_role"][q["role"]].append(q["resolved_id"])

        intersection = self._intersect_media_ids_across_constraints(
            state.responses, expected, intended_type)

        if not intersection:
            relaxed_state = self._relax_roles_and_retry_intersection(state)
            relaxed_intersection = self._intersect_media_ids_across_constraints(
                relaxed_state.responses, expected, intended_type)

            if not relaxed_intersection:
                fallback_step = FallbackHandler.generate_steps(
                    state.resolved_entities,
                    intents=state.extraction_result
                )
                if isinstance(fallback_step, dict):
                    fallback_step = [fallback_step]

                for fs in reversed(fallback_step):
                    state.plan_steps.insert(0, fs)

                from execution_orchestrator import ExecutionTraceLogger
                ExecutionTraceLogger.log_step(
                    step_id="fallback_injected_after_role_relaxation",
                    path="(internal)",
                    status="Fallback Injected",
                    summary="No matches after relaxing roles. Fallback discovery triggered.",
                    state=state
                )
                return state
            else:
                intersection = relaxed_intersection

        # 🚀 Inject lookup steps
        for item in intersection:
            if intended_type == "movie" or (intended_type == "both" and "title" in item):
                movie_id = item.get("id")
                if movie_id:
                    state.plan_steps.insert(0, {
                        "step_id": f"step_lookup_movie_{movie_id}",
                        "endpoint": f"/movie/{movie_id}",
                        "method": "GET",
                        "produces": [],
                        "requires": ["movie_id"]
                    })
            elif intended_type == "tv" or (intended_type == "both" and "name" in item):
                tv_id = item.get("id")
                if tv_id:
                    state.plan_steps.insert(0, {
                        "step_id": f"step_lookup_tv_{tv_id}",
                        "endpoint": f"/tv/{tv_id}",
                        "method": "GET",
                        "produces": [],
                        "requires": ["tv_id"]
                    })

        return state

    def _relax_roles_and_retry_intersection(self, state):
        """
        Relax stricter roles (director, writer, etc.) first.
        After dropping strict roles, retry intersection.
        Only if still no matches, reluctantly drop cast (actor) roles.
        """
        relaxed_roles = []
        if not hasattr(state, "relaxed_parameters"):
            state.relaxed_parameters = []

        # 1️⃣ Drop stricter crew roles first
        for role_prefix in ["step_director_", "step_writer_", "step_producer_", "step_composer_"]:
            for step_id in list(state.completed_steps):
                if step_id.startswith(role_prefix):
                    # print(f"♻️ Dropping step {step_id} to relax strict crew role constraint.")
                    state.completed_steps.remove(step_id)
                    state.data_registry.pop(step_id, None)
                    role_name = role_prefix.replace(
                        "step_", "").replace("_", "")

                    if not hasattr(state, "relaxed_parameters"):
                        state.relaxed_parameters = []
                    state.relaxed_parameters.append(role_name)

                    from execution_orchestrator import ExecutionTraceLogger
                    ExecutionTraceLogger.log_step(
                        step_id=step_id,
                        path="(internal)",
                        status="Role Relaxed",
                        summary=f"Dropped strict crew role step: {step_id}",
                        state=state
                    )

        media_type = getattr(state, "intended_media_type", "movie")
        media_key = "id"  # Both movie and TV credits use "id" for media

        results = []
        for step_id in state.completed_steps:
            step_data = state.data_registry.get(step_id, {})
            results.extend(step_data.get("cast", []))
            results.extend(step_data.get("crew", []))

        expected = {
            "person_ids": [
                e["resolved_id"]
                for e in state.extraction_result.get("query_entities", [])
                if e.get("type") == "person"
            ],
            "company_ids": state.resolved_entities.get("company_id", []),
            "network_ids": state.resolved_entities.get("network_id", [])
        }

        # 2️⃣ Retry intersection after dropping strict crew roles
        intersection = self._intersect_media_ids_across_constraints(
            results, expected, media_type)
        if isinstance(intersection, list) and len(intersection) > 0:
            # print(f"✅ Successful intersection after relaxing strict roles: {intersection}")
            return state

        # 3️⃣ If still no matches, reluctantly drop cast (actor) roles
        for step_id in list(state.completed_steps):
            if step_id.startswith("step_cast_"):
                # print(f"⚠️ Dropping step {step_id} (cast) to relax actor constraint.")
                state.completed_steps.remove(step_id)
                state.data_registry.pop(step_id, None)
                relaxed_roles.append("cast")
                from execution_orchestrator import ExecutionTraceLogger
                ExecutionTraceLogger.log_step(
                    step_id=step_id,
                    path="(internal)",
                    status="Role Relaxed (Cast)",
                    summary=f"Dropped actor step: {step_id}",
                    state=state
                )

        return state


class ExecutionTraceLogger:
    @staticmethod
    def log_step(step_id, path, status, summary=None, state=None):
        print("\n📍 Execution Trace")
        print(f"├─ Step: {step_id}")
        print(f"├─ Endpoint: {path}")
        print(f"├─ Status: {status}")

        print(
            f"🧾 Trace Entry Added → Step: {step_id}, Status: {status}, Notes: {summary}")

        # Format the result for print
        if summary is not None:
            try:
                text = summary if isinstance(
                    summary, str) else json.dumps(summary, default=str)
            except Exception:
                text = str(summary)
            print(f"└─ Result: {text[:100]}{'...' if len(text) > 100 else ''}")

        # Append to state trace
        if state is not None and hasattr(state, "execution_trace"):
            trace_entry = {
                "step_id": step_id,
                "endpoint": path,
                "status": status,
                "notes": summary if isinstance(summary, str) else str(summary),
                "constraint_tree": str(getattr(state, "constraint_tree", "")),
                "relaxation_log": list(getattr(state, "relaxation_log", [])),
                "injected_steps": [
                    getattr(s, "endpoint", str(s))
                    for s in getattr(state, "steps", [])
                ] if hasattr(state, "steps") else []
            }
            state.execution_trace.append(trace_entry)
# Usage inside orchestrator loop:
# After each response:
# will be moved inside try block where 'summaries' is defined

# On failure:
# will be moved inside exception block where 'response' is defined
