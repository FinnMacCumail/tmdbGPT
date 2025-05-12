from hashlib import sha256
import requests

from core.execution.post_validator import PostValidator
from core.execution.trace_logger import ExecutionTraceLogger
from core.execution.fallback import FallbackHandler, FallbackSemanticBuilder
from core.execution.discovery_handler import DiscoveryHandler
from core.planner.entity_reranker import EntityAwareReranker
from core.planner.plan_validator import PlanValidator, SymbolicConstraintFilter
from core.planner.dependency_manager import DependencyManager, inject_lookup_steps_from_role_intersection
from core.entity.param_utils import update_symbolic_registry
from core.formatting.registry import RESPONSE_RENDERERS
from core.formatting.templates import format_fallback, QueryExplanationBuilder
from core.model.constraint import Constraint
from response.log_summary import log_summary
from nlp.nlp_retriever import PostStepUpdater, PathRewriter, expand_plan_with_dependencies
from core.model.evaluator import evaluate_constraint_tree
from core.planner.constraint_planner import inject_validation_steps_from_ids
from core.entity.param_utils import enrich_symbolic_registry


class ExecutionOrchestrator:
    def __init__(self, base_url, headers):
        self.dependency_manager = DependencyManager()
        self.base_url = base_url
        self.headers = headers
        self.validator = PlanValidator()

    def execute(self, state):
        state.error = None
        state.data_registry = {}
        state.completed_steps = []
        seen_step_keys = set()
        step_origin_depth = {}
        MAX_CHAIN_DEPTH = 3

        print("🧪 About to execute steps:", [
              s["step_id"] for s in state.plan_steps])
        print("🎯 Intended media type:", state.intended_media_type)

        while state.plan_steps:
            step = state.plan_steps.pop(0)

            expanded = False
            for k, v in step.get("parameters", {}).items():
                if isinstance(v, list) and len(v) > 1 and f"{{{k}}}" in step.get("endpoint", ""):
                    for single_val in v:
                        new_step = step.copy()
                        new_step["parameters"] = {
                            **step["parameters"], k: single_val}
                        new_step["step_id"] = f"{step['step_id']}_{k}_{single_val}"
                        state.plan_steps.insert(0, new_step)
                    expanded = True
                    break
            if expanded:
                continue

            endpoint = step.get("endpoint")
            if not endpoint:
                continue

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

            missing_requires = [
                req for req in step.get("requires", [])
                if req not in state.resolved_entities
            ]

            if missing_requires:
                soft_filters = {"genre", "date", "runtime",
                                "votes", "rating", "language", "country"}
                soft_missing = [
                    req for req in missing_requires
                    if SymbolicConstraintFilter._map_key_to_entity(req) in soft_filters
                ]
                if soft_missing and len(soft_missing) == len(missing_requires):
                    step.setdefault("soft_relaxed", []).extend(soft_missing)
                else:
                    continue

            if step_id in state.completed_steps:
                continue

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

            params = step.get("parameters", {})
            if not isinstance(params, dict):
                params = {}

            path = step.get("endpoint")
            for k, v in params.items():
                if f"{{{k}}}" in path:
                    value = v[0] if isinstance(v, list) else v
                    path = path.replace(f"{{{k}}}", str(value))
            path = PathRewriter.rewrite(path, state.resolved_entities)
            full_url = f"{self.base_url}{path}"

            if isinstance(params.get("query"), dict):
                original = params["query"]
                params["query"] = original.get("name", "")

            param_string = "&".join(
                f"{k}={v}" for k, v in sorted(params.items()))
            dedup_key = f"{step['endpoint']}?{param_string}"
            step_hash = sha256(dedup_key.encode()).hexdigest()
            if step_hash in seen_step_keys:
                continue
            seen_step_keys.add(step_hash)

            try:
                response = requests.get(
                    full_url, headers=self.headers, params=params)
                if response.status_code == 200:
                    try:
                        json_data = response.json()
                        state.data_registry[step_id] = json_data
                        previous_entities = set(state.resolved_entities.keys())
                        state = PostStepUpdater.update(state, step, json_data)
                        new_entities = {
                            k: v for k, v in state.resolved_entities.items()
                            if k not in previous_entities
                        }

                        if step["endpoint"].startswith("/discover/movie"):
                            self._handle_discover_movie_step(
                                step, step_id, path, json_data, state, depth, seen_step_keys)
                        else:
                            DiscoveryHandler.handle_generic_response(
                                step, json_data, state)

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
                                state = inject_lookup_steps_from_role_intersection(
                                    state)

                        if new_entities:
                            new_steps = expand_plan_with_dependencies(
                                state, new_entities)
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

        format_type = state.response_format or "summary"
        renderer = RESPONSE_RENDERERS.get(format_type, format_fallback)
        final_output = renderer(state)
        state.formatted_response = final_output

        final_validated = []
        if getattr(state, "constraint_tree", None):
            for item in state.responses:
                if state.constraint_tree.is_satisfied_by(item):
                    final_validated.append(item)
            state.responses = final_validated

        if getattr(state, "satisfied_roles", None):
            roles_text = ", ".join(sorted(state.satisfied_roles))
            explanation_lines = [
                f"✅ Roles satisfied via intersection: {roles_text}."]
        else:
            explanation_lines = []

        state.explanation = "\n".join(
            explanation_lines + [
                QueryExplanationBuilder.build_final_explanation(
                    extraction_result=state.extraction_result,
                    relaxed_parameters=state.relaxed_parameters,
                    fallback_used=any(step.get("fallback_injected")
                                      for step in state.plan_steps)
                )
            ]
        )

        if getattr(state, "relaxed_parameters", []):
            relaxed_summary = ", ".join(sorted(set(state.relaxed_parameters)))
            ExecutionTraceLogger.log_step(
                "relaxation_summary",
                path="(global)",
                status="Relaxation Summary",
                summary=f"Relaxed constraints attempted before fallback: {relaxed_summary}",
                state=state
            )

        log_summary(state)
        return state

    def _handle_discover_movie_step(self, step, step_id, path, json_data, state, depth, seen_step_keys):
        filtered_movies = PostValidator.run_post_validations(
            step, json_data, state)

        if not filtered_movies:
            ExecutionTraceLogger.log_step(
                step_id, path, "Filtered", "No matching results", state=state
            )

            already_dropped = {p.strip()
                               for p in step_id.split("_relaxed_")[1:] if p}
            relaxed_steps = FallbackHandler.relax_constraints(
                step, already_dropped, state=state)

            if relaxed_steps:
                for relaxed_step in relaxed_steps:
                    if relaxed_step["step_id"] not in state.completed_steps:
                        constraint_dropped = relaxed_step["step_id"].split("_relaxed_")[
                            1]
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
                return

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
            state.completed_steps.append(step_id)
            return

        query_entities = state.extraction_result.get("query_entities", [])
        ranked = EntityAwareReranker.boost_by_entity_mentions(
            filtered_movies, query_entities)

        ids = evaluate_constraint_tree(
            state.constraint_tree, state.data_registry)

        # 🔍 Extract valid movie IDs from evaluated constraints
        valid_movie_ids = set()
        for id_group in ids.get("movie", {}).values():
            valid_movie_ids |= id_group

        matched_keys = set(ids.keys())
        matched = [f"{c.key}={c.value}" for c in state.constraint_tree if isinstance(
            c, Constraint) and c.key in matched_keys]
        relaxed = list(state.relaxation_log)
        validated = list(state.post_validation_log)

        if not hasattr(state, "satisfied_roles"):
            state.satisfied_roles = set()

        for movie in ranked:
            satisfied = movie.get("_provenance", {}).get("satisfied_roles", [])
            state.satisfied_roles.update(satisfied)

        for movie in ranked:
            movie_id = movie.get("id")
            if movie_id not in valid_movie_ids:
                continue  # 🔍 Skip movies that don't match constraints

            credits = None
            if movie_id:
                try:
                    res = requests.get(
                        f"{state.base_url}/movie/{movie_id}/credits",
                        headers=state.headers
                    )
                    if res.status_code == 200:
                        credits = res.json()
                except Exception as e:
                    print(
                        f"⚠️ Could not fetch credits for movie ID {movie_id}: {e}")

            movie["final_score"] = movie.get("final_score", 1.0)
            movie["type"] = "movie_summary"
            movie["_provenance"] = {
                "matched_constraints": matched,
                "relaxed_constraints": relaxed,
                "post_validations": validated
            }

            enrich_symbolic_registry(
                movie, state.data_registry, credits=credits)
            print(
                f"🧠 Appending validated movie: {movie.get('title')} with score {movie.get('final_score')}")
            state.responses.append(movie)

        state.data_registry[step_id]["validated"] = ranked

        if state.relaxation_log and (dropped := getattr(state, "last_dropped_constraints", [])):
            restored = []
            for c in dropped:
                ids = state.data_registry.get(
                    c.key, {}).get(str(c.value), set())
                if ids:
                    state.constraint_tree.constraints.append(c)
                    restored.append(f"{c.key}={c.value}")
            already_logged = set(state.relaxation_log)
            for restored_id in restored:
                msg = f"Restored: {restored_id}"
                if msg not in already_logged:
                    state.relaxation_log.append(msg)

        if not getattr(state, "constraint_tree_evaluated", False):
            ids = evaluate_constraint_tree(
                state.constraint_tree, state.data_registry)
            if ids:
                inject_validation_steps_from_ids(ids, state)
            state.constraint_tree_evaluated = True

        ExecutionTraceLogger.log_step(
            step_id, path, "Validated", summary=ranked[0], state=state
        )
        state.completed_steps.append(step_id)
