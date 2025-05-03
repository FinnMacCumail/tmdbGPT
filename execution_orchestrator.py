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
from post_validator import ResultScorer
from response_formatter import QueryExplanationBuilder
from plan_validator import SymbolicConstraintFilter
from constraint_model import evaluate_constraint_tree, relax_constraint_tree
from typing import TYPE_CHECKING
from dependency_manager import DependencyManager
from constraint_model import Constraint, ConstraintGroup
import hashlib
from pydantic import BaseModel


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
        movie_results = data.get("results", [])
        print(
            f"🔍 Running post-validations on {len(movie_results)} movie(s)...")

        for rule in self.VALIDATION_REGISTRY:
            if rule["endpoint"] in step["endpoint"] and rule["trigger_param"] in step.get("parameters", {}):
                print(
                    f"🧪 Applying validation rule: {rule['validator'].__name__}")
                validator = rule["validator"]
                build_args = rule["args_builder"]
                args = build_args(step, state)

                query_entities = state.extraction_result.get(
                    "query_entities", [])

                for movie in movie_results:
                    movie_id = movie.get("id")
                    if not movie_id:
                        continue

                    url = f"{self.base_url}/movie/{movie_id}/credits"
                    try:
                        response = requests.get(url, headers=self.headers)
                        if response.status_code != 200:
                            continue
                        result_data = response.json()

                        score = self._score_movie_against_query(
                            movie=movie,
                            credits=result_data,
                            step=step,
                            query_entities=query_entities
                        )

                        if score > 0:
                            movie["final_score"] = min(score, 1.0)
                            validated.append(movie)
                            print(
                                f"✅ Movie {movie_id} accepted with final score {movie['final_score']}")
                        else:
                            print(
                                f"❌ Movie {movie_id} rejected (no validations passed)")

                    except Exception as e:
                        print(
                            f"⚠️ Validation failed for movie_id={movie_id}: {e}")

                break  # Only apply the first matching rule

        return validated or movie_results

    def execute(self, state):
        print(f"\n[DEBUG] Entering Orchestrator Execution")
        print(f"🧭 [DEBUG] Initial question_type: {state.question_type}")
        print(f"🎨 [DEBUG] Initial response_format: {state.response_format}")

        state.error = None
        state.data_registry = {}
        state.completed_steps = []
        seen_step_keys = set()
        step_origin_depth = {}
        MAX_CHAIN_DEPTH = 3

        print(f"🧭 Question Type: {getattr(state, 'question_type', None)}")
        print(f"🎨 Response Format: {getattr(state, 'response_format', None)}")

        # ✅ Phase 21.5: Evaluate and inject constraint-tree-based steps (with relaxation fallback)
        if self._evaluate_and_inject_from_constraint_tree(state):
            print("✅ Constraint tree steps injected.")
        else:
            print("🛑 No executable steps from constraint tree.")

        # ✅ phase 9.2 - pgpv - Safety check happens AFTER constraint planning
        if not self._safe_to_execute(state):
            print(f"🛑 Fallback triggered due to unsafe plan.")
            return state

        while state.plan_steps:
            step = state.plan_steps.pop(0)  # process from front

            # phase 19.9 - Media Type Enforcement Baseline
            if state.intended_media_type and step.get("endpoint"):
                if state.intended_media_type != "both":
                    if "/tv" in step["endpoint"] and state.intended_media_type != "tv":
                        print(
                            f"⏭️ Skipping TV step {step['step_id']} for movie query.")
                        continue
                    if "/movie" in step["endpoint"] and state.intended_media_type != "movie":
                        print(
                            f"⏭️ Skipping Movie step {step['step_id']} for TV query.")
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
                    print(
                        f"⚡ Soft relaxation: missing only soft filters {soft_missing}. Proceeding with relaxed step.")
                    # ✅ Mark the step as relaxed so post-filtering can occur later
                    step.setdefault("soft_relaxed", []).extend(soft_missing)
                else:
                    print(
                        f"⏭️ Skipping step {step_id}: missing required core entities {missing_requires}")
                    continue  # Skip hard requirements

            print(f"\n[DEBUG] Executing Step: {step_id}")
            print(f"[DEBUG] Current question_type: {state.question_type}")
            print(f"[DEBUG] Current response_format: {state.response_format}")

            print(f"▶️ Popped step: {step_id}")
            print(
                f"🧾 Queue snapshot (after pop): {[s['step_id'] for s in state.plan_steps]}")
            if not state.plan_steps:
                if not step.get("fallback_injected"):  # ✅ NEW: avoid fallback looping
                    state = DependencyManager.analyze_dependencies(state)
                    # 🚀 NEW: inject lookup steps after role-based intersection
                    state = self._inject_lookup_steps_from_role_intersection(
                        state)
                else:
                    print(
                        f"🛑 Step {step.get('step_id')} is fallback-injected. Skipping dependency expansion and lookup injection.")

            if step_id in state.completed_steps:
                print(f"✅ Skipping already completed step: {step_id}")
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
                print(
                    f"🔁 Loop suppression: skipping step {step_id} (depth={depth})")
                continue

            # 🛡 Sanity check on parameters
            params = step.get("parameters", {})
            if not isinstance(params, dict):
                print(
                    f"🚨 Malformed parameters in step {step_id} → {type(params)}")
                params = {}
            else:
                assert isinstance(
                    params, dict), f"❌ Step {step_id} has non-dict parameters: {type(params)}"

            # 🧠 Replace placeholders in the path using updated params
            path = step.get("endpoint")
            for k, v in params.items():
                if f"{{{k}}}" in path:
                    # Fix: handle list injection
                    value = v[0] if isinstance(v, list) else v
                    path = path.replace(f"{{{k}}}", str(value))
                    print(f"🧩 Replaced path slot: {{{k}}} → {v}")
            print(f"🛠️ Resolved full path: {path}")
            path = PathRewriter.rewrite(path, state.resolved_entities)
            full_url = f"{self.base_url}{path}"
            print(f"\n⚡ Executing {step_id}: {path}")

            # Sanitize structured query parameter
            if isinstance(params.get("query"), dict):
                original = params["query"]
                params["query"] = original.get("name", "")
                print(
                    f"🔧 Flattened structured query param from {original} → '{params['query']}'")

            # ✅ Deduplication AFTER path + param injection
            param_string = "&".join(
                f"{k}={v}" for k, v in sorted(params.items()))
            dedup_key = f"{step['endpoint']}?{param_string}"
            step_hash = sha256(dedup_key.encode()).hexdigest()

            if step_hash in seen_step_keys:
                print(
                    f"🔁 Skipping duplicate step_id {step_id} (hash={step_hash}) with same parameters")
                continue

            seen_step_keys.add(step_hash)

            try:
                print(f"📤 Calling TMDB: {full_url}")
                print(f"📦 Params: {params}")
                response = requests.get(
                    full_url, headers=self.headers, params=params)

                if response.status_code == 200:
                    print(f"✅ Success: {response.status_code}")
                    try:
                        json_data = response.json()
                        state.data_registry[step_id] = json_data

                        print(
                            f"📦 Registry contents after {step_id}: {state.data_registry.get(step_id)}")

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

                        # Append new steps if needed
                        if new_entities:
                            new_steps = expand_plan_with_dependencies(
                                state, new_entities)
                            if new_steps:
                                print(
                                    f"🔁 Appending {len(new_steps)} new dependent step(s) to execution queue.")
                                for new_step in new_steps:
                                    state.plan_steps.append(new_step)
                                    step_origin_depth[new_step["step_id"]
                                                      ] = depth + 1

                    except Exception as ex:
                        print(f"⚠️ Could not parse JSON or update state: {ex}")
            except Exception as ex:
                print(f"🔥 Step {step_id} failed with exception: {ex}")
                ExecutionTraceLogger.log_step(
                    step_id, path, f"Failed ({str(ex)})", state=state)
                state.error = str(ex)

        # 👇 Safely determine the format type from state
        # format_type = getattr(state, "response_format", "summary")
        format_type = state.response_format or "summary"
        renderer = RESPONSE_RENDERERS.get(format_type, format_fallback)

        # 👇 Generate final formatted output
        final_output = renderer(state)

        print("\n--- FINAL RESPONSE ---")
        print(final_output)

        # 👇 You can optionally assign it to state if needed
        state.formatted_response = final_output

        state.explanation = QueryExplanationBuilder.build_final_explanation(
            extraction_result=state.extraction_result,
            relaxed_parameters=state.relaxed_parameters,
            fallback_used=any(step.get("fallback_injected")
                              for step in state.plan_steps)
        )

        print("[DEBUG] Orchestrator execution completed.")
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

        return state

    def _score_movie_against_query(self, movie, query_entities, constraint_tree, state):
        provenance = {
            "matched_constraints": [],
            "relaxed_constraints": getattr(state, "relaxation_log", []),
            "post_validations": []
        }

        score = 0.0

        # 🎯 1. Post-validation checks (already implemented)
        if self.post_validator.has_all_cast(movie, query_entities):
            score += 0.5
            provenance["post_validations"].append("has_all_cast")
        if self.post_validator.has_director(movie, query_entities):
            score += 0.3
            provenance["post_validations"].append("has_director")
        if self.post_validator.validate_genres(movie, query_entities):
            score += 0.2
            provenance["post_validations"].append("validate_genres")

        # 🎯 2. Constraint matches from constraint_tree
        def collect_constraints(group):
            for c in group:
                if isinstance(c, ConstraintGroup):
                    yield from collect_constraints(c)
                elif isinstance(c, Constraint):
                    yield c

        for constraint in collect_constraints(constraint_tree):
            # e.g., constraint.key = 'with_genres', constraint.value = 18
            movie_val = movie.get(constraint.key)
            if movie_val:
                if isinstance(movie_val, list) and constraint.value in movie_val:
                    provenance["matched_constraints"].append(
                        f"{constraint.key}={constraint.value}")
                elif movie_val == constraint.value:
                    provenance["matched_constraints"].append(
                        f"{constraint.key}={constraint.value}")

        movie["_provenance"] = provenance
        return score

    # phase 21.5 - ID injection logic
    def _inject_validation_steps_from_ids(self, ids_by_type: Dict[str, Set[int]], state):
        validation_steps = []
        for media_type, id_set in ids_by_type.items():
            for id_ in sorted(id_set):
                param_key = f"{media_type}_id"
                validation_steps.append({
                    "step_id": f"step_validate_{media_type}_{id_}",
                    "endpoint": f"/{media_type}/{id_}",
                    "method": "GET",
                    "requires": [param_key],
                    "params": {param_key: id_},  # ✅ added this
                    "produces": [],
                    "from_constraint_tree": True
                })
        print(
            f"✅ Injecting {len(validation_steps)} validation steps from constraint tree.")
        print(f"📦 Validation steps injected: {validation_steps}")
        state.plan_steps = validation_steps + state.plan_steps

    # phase 21.5 - Constraint-aware fallback / relaxation

    def _evaluate_and_inject_from_constraint_tree(self, state) -> bool:
        """
        Evaluates the constraint tree and injects validation steps if any matches are found.
        If no matches, performs a relaxation and retries once.

        Returns True if any steps were injected, else False.
        """
        fingerprint = self._make_constraint_fingerprint(state.constraint_tree)

        if fingerprint in state.visited_fingerprints:
            print(
                f"🔁 [PRUNE] Skipping already visited fingerprint: {fingerprint}")
            print(f"🧾 [Visited Fingerprints]: {state.visited_fingerprints}")

            ExecutionTraceLogger.log_step(
                step_id="step_pruning",
                path="(internal)",
                status="Skipped execution due to repeated constraint fingerprint",
                summary={"fingerprint": fingerprint},
                state=state
            )
            return False

        # ✅ First-time fingerprint
        state.visited_fingerprints.add(fingerprint)
        print(f"🧪 [NEW Fingerprint Recorded]: {fingerprint}")

        if not hasattr(state, "constraint_tree") or not state.constraint_tree:
            print("⛔ No constraint tree available on state.")
            return False

        print("🔍 Evaluating constraint tree...")
        print(f"🌲 Tree structure: {state.constraint_tree}")

        print(
            f"🔎 Current data_registry keys: {list(state.data_registry.keys())}")
        print(
            f"🔎 Registry[with_genres]: {state.data_registry.get('with_genres')}")

        ids = evaluate_constraint_tree(
            state.constraint_tree, state.data_registry)

        print(f"🎯 Constraint evaluation returned: {ids}")
        if ids:
            print("✅ Injecting validation steps from constraint matches...")
            self._inject_validation_steps_from_ids(ids, state)
            return True

        print("⚠️ Phase 21.5 - No matches. Attempting constraint-based relaxation...")

        relaxed_tree, dropped, reasons = relax_constraint_tree(
            state.constraint_tree, max_drops=1
        )

        if reasons:
            print(f"🧾 Relaxation reasons: {reasons}")
            state.relaxation_log.extend(reasons)

        if dropped:
            for constraint in dropped:
                reason = f"Dropped '{constraint.key}={constraint.value}' (priority={constraint.priority}, confidence={constraint.confidence})"
                print(f"📝 {reason}")
                state.relaxation_log.append(reason)

        if relaxed_tree:
            print(f"♻️ Relaxed constraint tree: {relaxed_tree}")
            relaxed_ids = evaluate_constraint_tree(
                relaxed_tree, state.data_registry)
            print(f"🎯 Relaxed constraint evaluation returned: {relaxed_ids}")
            if relaxed_ids:
                print("✅ Injecting validation steps from relaxed constraint matches...")
                self._inject_validation_steps_from_ids(relaxed_ids, state)
                state.constraint_tree = relaxed_tree
                return True
            else:
                print("🛑 Still no matches after relaxing constraints.")
        else:
            print("🛑 Cannot relax further — no constraints left.")

        print("🛑 No validation steps injected from constraint tree.")
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
        print(f"🔎 BEGIN _handle_discover_movie_step for {step_id}")

        filtered_movies = self._run_post_validations(step, json_data, state)

        if filtered_movies:
            print(f"✅ Found {len(filtered_movies)} filtered result(s)")
            query_entities = state.extraction_result.get("query_entities", [])
            ranked = EntityAwareReranker.boost_by_entity_mentions(
                filtered_movies, query_entities)
            state.data_registry[step_id]["validated"] = ranked

            print(
                f"📦 Registry contents after {step_id}: {state.data_registry.get(step_id)}")

            matched = [c.to_string()
                       for c in state.constraint_tree if hasattr(c, "to_string")]
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
                state.responses.append(movie)

            ExecutionTraceLogger.log_step(
                step_id, path, "Validated", summary=ranked[0], state=state
            )
            state.completed_steps.append(step_id)
            print(f"✅ Step marked completed: {step_id}")
            return

        # ❌ No valid results — trigger relaxation
        print("⚠️ No valid results matched required cast/director.")
        ExecutionTraceLogger.log_step(
            step_id, path, "Filtered", "No matching results", state=state
        )
        state.responses.append(
            "⚠️ No valid results matched all required cast/director.")

        already_dropped = set()
        if "_relaxed_" in step_id:
            already_dropped.update(p.strip()
                                   for p in step_id.split("_relaxed_")[1:] if p)

        relaxed_steps = FallbackHandler.relax_constraints(
            step, already_dropped, state=state)

        if relaxed_steps:
            for relaxed_step in relaxed_steps:
                if relaxed_step["step_id"] not in state.completed_steps:
                    constraint_dropped = relaxed_step["step_id"].split("_relaxed_")[
                        1]
                    print(
                        f"♻️ Injected relaxed retry: {relaxed_step['step_id']} (Dropped {constraint_dropped})")
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
            print(f"✅ Marked original step completed after injecting relaxed retries.")
            return

        # 🛑 No more relaxation possible → fallback
        print("🛑 All filter drop retries exhausted. Injecting semantic fallback...")

        fallback_step = FallbackSemanticBuilder.enrich_fallback_step(
            original_step=step,
            extraction_result=state.extraction_result,
            resolved_entities=state.resolved_entities
        )

        if fallback_step["step_id"] not in state.completed_steps:
            state.plan_steps.insert(0, fallback_step)
            ExecutionTraceLogger.log_step(
                fallback_step["step_id"],
                path=fallback_step["endpoint"],
                status="Semantic Fallback Injected",
                summary=f"Enriched fallback injected with parameters: {fallback_step.get('parameters', {})}",
                state=state
            )
            print(
                f"🧭 Injected enriched fallback step: {fallback_step['endpoint']}")
        else:
            print("⚠️ Fallback already completed — skipping reinjection.")

        state.completed_steps.append(step_id)
        print(f"✅ Marked as completed: {step_id}")

    def _handle_generic_response(self, step, step_id, path, json_data, state):
        print(f"📥 Handling generic response for {path}...")

        summaries = ResultExtractor.extract(
            path, json_data, state.resolved_entities)
        print(
            f"🔎 ResultExtractor.extract returned {len(summaries)} summaries for endpoint: {path}")

        query_entities = state.extraction_result.get("query_entities", [])
        role_tagged = any(e.get("role") for e in query_entities)

        # ✅ Always apply fallback tagging first
        if step.get("fallback_injected") and isinstance(json_data, dict) and "results" in json_data:
            print(
                f"♻️ Tagging fallback-injected results from {step['endpoint']}")
            for movie in json_data["results"]:
                movie["final_score"] = 0.3
                movie["source"] = step["endpoint"] + "_relaxed"

        # ✅ Post-filter extracted summaries
        if summaries:
            filtered_summaries = ResultExtractor.post_filter_responses(
                summaries,
                query_entities=query_entities,
                extraction_result=state.extraction_result
            )
            print(
                f"🔎 Post-filtered to {len(filtered_summaries)} summaries after entity matching")
            summaries = filtered_summaries

        # 🎯 NEW: Phase 20.4 — Role Validation for each summary
        validated_summaries = []
        for summary in summaries:
            validations = ResultScorer.validate_entity_matches(
                summary, query_entities)
            score = ResultScorer.score_matches(validations)
            summary["final_score"] = max(summary.get("final_score", 0), score)

            if summary["final_score"] >= 0.5:  # Only accept reasonable matches
                validated_summaries.append(summary)
                print(
                    f"🎯 Validated {summary.get('title', 'Unknown')} → Score: {summary['final_score']}")
            else:
                print(
                    f"⚠️ Low score ({summary['final_score']}) for {summary.get('title', 'Unknown')} — skipping.")

        # 🛡 Optional: if no validated results, fallback
        if not validated_summaries:
            print(
                f"🛑 No high-quality results after validation for {step_id}. Injecting fallback...")

            fallback_step = FallbackSemanticBuilder.enrich_fallback_step(
                original_step=step,
                extraction_result=state.extraction_result,
                resolved_entities=state.resolved_entities
            )

            if fallback_step["step_id"] not in state.completed_steps:
                state.plan_steps.insert(0, fallback_step)
                print(
                    f"🧭 Injected enriched fallback step: {fallback_step['endpoint']}")

            state.completed_steps.append(step_id)
            return  # Stop handling this batch

        # ✅ Append validated summaries
        state.responses.extend(validated_summaries)

        # ✅ Log completion
        ExecutionTraceLogger.log_step(
            step_id, path, "Handled", validated_summaries[:1] if validated_summaries else [], state=state)
        state.completed_steps.append(step_id)
        print(f"✅ Step marked completed: {step_id}")

    def _intersect_movie_ids_across_roles(self, state) -> dict:
        """
        Intersect movie IDs and additional constraints (company/network) across completed steps.
        Returns:
            dict with "movie_ids" and "tv_ids"
        """
        movie_sets = []
        tv_sets = []
        company_sets = []
        network_sets = []

        for step_id in state.completed_steps:
            result = state.data_registry.get(step_id, {})
            if not isinstance(result, dict):
                continue

            if step_id.startswith("step_cast_") or step_id.startswith("step_director_") or step_id.startswith("step_writer_") or step_id.startswith("step_producer_") or step_id.startswith("step_composer_"):
                ids = {m.get("id") for m in result.get(
                    "cast", []) + result.get("crew", []) if m.get("id")}
                if ids:
                    movie_sets.append(ids)

            elif step_id.startswith("step_company_"):
                ids = {m.get("id")
                       for m in result.get("results", []) if m.get("id")}
                if ids:
                    movie_sets.append(ids)
                    company_ids = {
                        company.get("id") for m in result.get("results", [])
                        for company in m.get("production_companies", [])
                        if company.get("id")
                    }
                    if company_ids:
                        company_sets.append(company_ids)

            elif step_id.startswith("step_network_"):
                ids = {m.get("id")
                       for m in result.get("results", []) if m.get("id")}
                if ids:
                    tv_sets.append(ids)
                    network_ids = {
                        network.get("id") for m in result.get("results", [])
                        for network in m.get("networks", [])
                        if network.get("id")
                    }
                    if network_ids:
                        network_sets.append(network_ids)

        # 🔁 Intersect movies by all role + company constraints
        intersected_movie_ids = set.intersection(
            *movie_sets) if movie_sets else set()
        if company_sets:
            intersected_movie_ids &= set.intersection(*company_sets)

        # 🔁 Intersect TV shows by all role + network constraints
        intersected_tv_ids = set.intersection(*tv_sets) if tv_sets else set()
        if network_sets:
            intersected_tv_ids &= set.intersection(*network_sets)

        print(f"🎯 Intersected movie IDs: {intersected_movie_ids}")
        print(f"🎯 Intersected TV IDs: {intersected_tv_ids}")
        return {
            "movie_ids": intersected_movie_ids,
            "tv_ids": intersected_tv_ids
        }

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
        print(
            f"✅ Injecting {len(validation_steps)} validation step(s) after intersection.")
        state.plan_steps = validation_steps + state.plan_steps

    def _safe_to_execute(self, state) -> bool:
        if not state.plan_steps:
            print(f"🛑 No steps available to execute — fallback needed.")
            return False

        # ✅ NEW: allow if any /discover/ steps present
        media_steps = [
            step for step in state.plan_steps
            if "/discover/" in (step.get("endpoint") or "")
        ]

        if media_steps:
            print(
                f"⚡ Proceeding with {len(media_steps)} discovery step(s): {[s['endpoint'] for s in media_steps]}")
            return True

        if len(state.plan_steps) == 1:
            step = state.plan_steps[0]
            produces = step.get("produces", [])
            if SymbolicConstraintFilter.is_media_endpoint(produces):
                print(
                    f"⚡ Proceeding with single media-producing step: {step['step_id']} ({step['endpoint']})")
                return True

        # 🧠 Otherwise: try intersection
        intersected_ids = self._intersect_movie_ids_across_roles(state)
        if intersected_ids:
            self._inject_validation_steps(state, intersected_ids)
            return True

        print(
            f"🛑 [Fallback Trigger] No executable steps in plan. Current steps: {state.plan_steps}")
        print(f"🛑 No intersection or valid steps — fallback needed.")
        return False

    def _inject_lookup_steps_from_role_intersection(self, state):
        """
        After dependency steps (credits) are completed,
        intersect movies/tv across roles,
        inject lookup steps accordingly,
        or relax roles if needed,
        or fallback gracefully if still empty.
        """
        intersection = self._intersect_movie_ids_across_roles(state)
        intended_type = getattr(state, "intended_media_type", "both") or "both"

        found_movies = intersection["movie_ids"]
        found_tv = intersection["tv_ids"]

        if not found_movies and not found_tv:
            # 🚨 No intersection — try relaxing role constraints first
            print("⚠️ No intersection found. Attempting to relax roles...")
            relaxed_state = self._relax_roles_and_retry_intersection(state)

            # After relaxing, retry intersection
            relaxed_intersection = self._intersect_movie_ids_across_roles(
                relaxed_state)
            found_movies = relaxed_intersection["movie_ids"]
            found_tv = relaxed_intersection["tv_ids"]

            if not found_movies and not found_tv:
                # 🚨 Still nothing — trigger fallback
                print("🛑 No matches after relaxing roles. Triggering fallback...")

                fallback_step = FallbackHandler.generate_steps(
                    state.resolved_entities,
                    intents=state.extraction_result
                )

                if isinstance(fallback_step, dict):
                    fallback_step = [fallback_step]

                for fs in reversed(fallback_step):
                    print(f"♻️ Injected fallback step: {fs.get('endpoint')}")
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
                # ✅ Intersection successful after relaxing
                print(
                    f"✅ Found intersection after relaxing roles: {found_movies or found_tv}")

        # 🚀 Inject lookup steps
        if intended_type == "movie":
            for movie_id in sorted(found_movies):
                lookup_step = {
                    "step_id": f"step_lookup_movie_{movie_id}",
                    "endpoint": f"/movie/{movie_id}",
                    "method": "GET",
                    "produces": [],
                    "requires": ["movie_id"]
                }
                print(f"🔎 Injected movie lookup step: {lookup_step}")
                state.plan_steps.insert(0, lookup_step)

        elif intended_type == "tv":
            for tv_id in sorted(found_tv):
                lookup_step = {
                    "step_id": f"step_lookup_tv_{tv_id}",
                    "endpoint": f"/tv/{tv_id}",
                    "method": "GET",
                    "produces": [],
                    "requires": ["tv_id"]
                }
                print(f"🔎 Injected tv lookup step: {lookup_step}")
                state.plan_steps.insert(0, lookup_step)

        elif intended_type == "both":
            if found_movies:
                for movie_id in sorted(found_movies):
                    lookup_step = {
                        "step_id": f"step_lookup_movie_{movie_id}",
                        "endpoint": f"/movie/{movie_id}",
                        "method": "GET",
                        "produces": [],
                        "requires": ["movie_id"]
                    }
                    print(f"🔎 Injected movie lookup step: {lookup_step}")
                    state.plan_steps.insert(0, lookup_step)
            elif found_tv:
                for tv_id in sorted(found_tv):
                    lookup_step = {
                        "step_id": f"step_lookup_tv_{tv_id}",
                        "endpoint": f"/tv/{tv_id}",
                        "method": "GET",
                        "produces": [],
                        "requires": ["tv_id"]
                    }
                    print(f"🔎 Injected tv lookup step: {lookup_step}")
                    state.plan_steps.insert(0, lookup_step)

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
                    print(
                        f"♻️ Dropping step {step_id} to relax strict crew role constraint.")
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

        # 2️⃣ Retry intersection after dropping strict crew roles
        intersection = self._intersect_movie_ids_across_roles(state)
        if intersection["movie_ids"] or intersection["tv_ids"]:
            print(
                f"✅ Successful intersection after relaxing strict roles: {intersection}")
            return state

        # 3️⃣ If still no matches, reluctantly drop cast (actor) roles
        for step_id in list(state.completed_steps):
            if step_id.startswith("step_cast_"):
                print(
                    f"⚠️ Dropping step {step_id} (cast) to relax actor constraint.")
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
