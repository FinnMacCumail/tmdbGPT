from typing import List, Dict, Any
from hashlib import sha256
import requests
import copy

from .trace_logger import ExecutionTraceLogger
from core.execution.fallback import FallbackHandler
from core.execution.discovery_handler import DiscoveryHandler
from core.formatting.registry import RESPONSE_RENDERERS
from core.formatting.templates import format_fallback, QueryExplanationBuilder
from core.planner.plan_utils import filter_valid_movies_or_tv
from core.entity.param_utils import enrich_symbolic_registry
from core.model.constraint import Constraint
from core.model.evaluator import evaluate_constraint_tree
from core.planner.constraint_planner import inject_validation_steps_from_ids
from core.planner.plan_validator import PlanValidator, contains_person_role_constraints
from nlp.nlp_retriever import PostStepUpdater, PathRewriter, expand_plan_with_dependencies
from core.planner.dependency_manager import DependencyManager, inject_lookup_steps_from_role_intersection

from response.log_summary import log_summary
from core.validation.role_validators import validate_roles, score_role_validation
from core.execution.fallback import FallbackSemanticBuilder


class StepRunner:
    def __init__(self, base_url: str, headers: Dict[str, str]):
        self.base_url = base_url
        self.headers = headers
        self.validator = PlanValidator()

    def execute(self, state):
        state.error = None
        if not hasattr(state, "data_registry"):
            state.data_registry = {}
        seen_step_keys = set()
        state.completed_steps = []
        step_origin_depth = {}
        MAX_CHAIN_DEPTH = 3

        if not hasattr(state, "roles_injected"):
            state.roles_injected = False

        while state.plan_steps:
            step = state.plan_steps.pop(0)
            # 🔀 Expands a step into multiple single-valued steps if a path parameter contains a list.
            # Enables correct execution of endpoints like /person/{person_id} for each value individually. ie - "person_id": [123, 456]
            expanded_steps = self._expand_multi_value_params(step)
            for step in expanded_steps:
                step_id = step.get("step_id")

                endpoint = step.get("endpoint")
                if not endpoint:
                    continue
                # prevents duplicate execution of steps that have already been run
                if step_id in state.completed_steps:
                    continue

                depth = step_origin_depth.get(step_id, 0)
                if depth > MAX_CHAIN_DEPTH:
                        f"⚠️ Skipping step {step_id} due to max chain depth.")
                    continue

                endpoint = step.get("endpoint")

                # 🛑 Pause discovery until all role credit steps are completed (if query has roles)
                if endpoint.startswith("/discover/") and contains_person_role_constraints(getattr(state, "constraint_tree", None)):
                    expected_role_steps = set()
                    for qe in state.extraction_result.get("query_entities", []):
                        if qe.get("type") == "person" and qe.get("role") and "resolved_id" in qe:
                            role = qe["role"]
                            pid = qe["resolved_id"]
                            media_type = getattr(
                                state, "intended_media_type", "movie")
                            if media_type == "both":
                                expected_role_steps.update({
                                    f"step_{role}_{pid}_tv",
                                    f"step_{role}_{pid}_movie"
                                })
                            else:
                                expected_role_steps.add(
                                    f"step_{role}_{pid}_{media_type}")
                    if not expected_role_steps.issubset(set(state.completed_steps)):
                        missing = expected_role_steps - \
                            set(state.completed_steps)
                            f"⏸️ Discovery step '{step['step_id']}' deferred — waiting on role credit steps: {missing}")
                        state.plan_steps.append(step)
                        continue

                # 🔍 Media-Type Filtering (TV vs Movie)
                # 🛑 Skip steps that target the wrong media type (e.g., TV step in a movie query).
                # Ensures query stays consistent with the user's intended content type
                if self._should_skip_step_based_on_media_type(step, state):
                    continue

                # 🛡️ Missing Entity Validation
                # 🚫 Skip step if required entities (e.g., person, network) are unresolved.
                # Soft-missing fields like genre/date are marked for potential fallback instead.
                if self._has_missing_required_entities(step, state):
                        f"⏭️ Skipping step {step.get('step_id')} — endpoint '{step.get('endpoint')}' does not match intended media type: {state.intended_media_type}")
                    continue

                # Inject path slot parameters
                # 🧩 Inject resolved entity values into path parameters (e.g., {person_id} → 123),
                # but skip relaxed fallback steps to avoid overriding their predefined values.
                if "_relaxed" not in step.get("step_id", ""):
                    step = self.validator.inject_path_slot_parameters(
                        step,
                        resolved_entities=state.resolved_entities,
                        extraction_result=state.extraction_result
                    )
                # 🧩 Final rewrite of endpoint path using current resolved entities.
                # Ensures any missed or newly added slot values are correctly substituted before request.
                path = PathRewriter.rewrite(endpoint, state.resolved_entities)
                full_url = f"{self.base_url}{path}"

                # Deduplication hash based on parameter set
                params = step.get("parameters", {})
                if not isinstance(params, dict):
                    params = {}

                # ✅ Normalize nested query param
                if isinstance(params.get("query"), dict):
                    original = params["query"]
                    params["query"] = original.get("name", "")

                # 🔐 Generate a deduplication hash from endpoint and parameters.
                # Skips execution if this exact step (same URL + params) has already been seen.
                param_string = "&".join(
                    f"{k}={v}" for k, v in sorted(params.items()))
                dedup_key = f"{endpoint}?{param_string}"
                step_hash = sha256(dedup_key.encode()).hexdigest()
                if step_hash in seen_step_keys:
                    continue
                seen_step_keys.add(step_hash)

                try:
                    response = requests.get(
                        full_url, headers=self.headers, params=params)
                    if response.status_code == 200:
                        json_data = response.json()

                        # 📦 Cache the raw response
                        state.data_registry[step_id] = json_data

                        # 🧠 Snapshot resolved entities before update to detect which new entities are added by this step.
                        previous_entities = set(state.resolved_entities.keys())
                        state = PostStepUpdater.update(state, step, json_data)
                        # ➕ Identify newly resolved entities to trigger follow-up planning (e.g., genre → inject discovery step).
                        new_entities = {
                            k: v for k, v in state.resolved_entities.items()
                            if k not in previous_entities
                        }

                        if endpoint.startswith("/discover/"):
                            DiscoveryHandler.handle_discover_step(
                                step, step_id, path, json_data, state, depth, seen_step_keys)
                        else:
                            DiscoveryHandler.handle_result_step(
                                step, json_data, state)

                        # 🔗 After all symbolic role steps (e.g., cast, director) are complete, trigger intersection.
                        # Injects /movie/{id} or /tv/{id} lookups for results satisfying all role-based constraints.
                        # Skips if fallback has already been applied to avoid redundant logic.
                        # Ensures queries like "movies directed by X and starring Y" don’t prematurely fallback.
                        if endpoint.startswith("/discover/") and not step.get("fallback_injected"):
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
                                    "✅ All symbolic role steps complete → triggering intersection")
                                state = DependencyManager.analyze_dependencies(
                                    state)
                                state = inject_lookup_steps_from_role_intersection(
                                    state)

                        self._inject_role_steps_if_ready(state)

                        # 🔁 After updating state, detect newly resolved entities and expand dependent steps.
                        for new_step in expand_plan_with_dependencies(state, new_entities):
                            state.plan_steps.append(new_step)
                            step_origin_depth[new_step["step_id"]] = depth + 1

                except Exception as ex:
                    ExecutionTraceLogger.log_step(
                        step_id, path or endpoint, f"Failed: {ex}", state=state)
                    state.error = str(ex)

                state.completed_steps.append(step_id)

        # 🧹 Apply symbolic filtering and deduplicate final results by ID, title, or name.
        # Ensures clean, constraint-compliant output without redundant entries.
        return self.finalize(state)

    def _inject_role_steps_if_ready(self, state):
        if state.roles_injected:
            return

        expected_role_steps = {
            f"step_{qe['role']}_{qe['resolved_id']}"
            for qe in state.extraction_result.get("query_entities", [])
            if qe.get("type") == "person" and qe.get("role")
        }
        if expected_role_steps.issubset(set(state.completed_steps)):
            state = DependencyManager.analyze_dependencies(state)
            state = inject_lookup_steps_from_role_intersection(state)
            state.roles_injected = True

    def _should_skip_step_based_on_media_type(self, step, state) -> bool:
        endpoint = step.get("endpoint", "")
        resolved_path = PathRewriter.rewrite(endpoint, state.resolved_entities)
        if state.intended_media_type == "tv" and "/movie" in resolved_path:
            return True
        if state.intended_media_type == "movie" and "/tv" in resolved_path:
            return True
        return False

    def _has_missing_required_entities(self, step, state) -> bool:
        missing = [r for r in step.get(
            "requires", []) if r not in state.resolved_entities]
        soft_filters = {"genre", "date", "runtime",
                        "votes", "rating", "language", "country"}
        soft_missing = [r for r in missing if r in soft_filters]
        if soft_missing and len(soft_missing) == len(missing):
            step.setdefault("soft_relaxed", []).extend(soft_missing)
        return bool(missing) and not (missing == soft_missing)

    def _expand_multi_value_params(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 🔁 Path Parameter Expansion Safeguards
        for k, v in step.get("parameters", {}).items():
            if isinstance(v, list) and len(v) > 1 and f"{{{k}}}" in step.get("endpoint", ""):
                expanded = []
                for val in v:
                    new_step = copy.deepcopy(step)
                    new_step["parameters"][k] = val
                    new_step["step_id"] = f"{step['step_id']}_{k}_{val}"
                    expanded.append(new_step)
                return expanded
        return [step]

    def finalize(self, state):
        original_count = len(state.responses)
        constraint_tree = getattr(state, "constraint_tree", None)
        data_registry = state.data_registry

        # ✅ Symbol-aware or symbol-free filtering
        if constraint_tree:
            filtered = [
                r for r in filter_valid_movies_or_tv(state.responses, constraint_tree, data_registry)
                if r.get("final_score", 0) > 0
            ]
        else:
            filtered = [r for r in state.responses if r.get(
                "final_score", 0) > 0]

        # 🧹 Deduplicate
        seen, deduped = set(), []
        for r in filtered:
            key = r.get("id") or r.get("title") or r.get("name")
            if key and key not in seen:
                deduped.append(r)
                seen.add(key)
        state.responses = deduped

        # 🆘 Inject fallback if nothing remains
        if not state.responses:
                "⚠️ No valid responses after filtering and deduplication — injecting final fallback")
            fallback_step = FallbackSemanticBuilder.enrich_fallback_step(
                original_step={"endpoint": "/discover/movie"},
                extraction_result=state.extraction_result,
                resolved_entities=state.resolved_entities
            )
            if fallback_step["step_id"] not in state.completed_steps:
                state.plan_steps.insert(0, fallback_step)
                ExecutionTraceLogger.log_step(
                    fallback_step["step_id"], fallback_step["endpoint"],
                    status="Final Fallback Injected",
                    summary="No valid results — fallback enrichment triggered",
                    state=state
                )

        # 📊 Deduplication log
        ExecutionTraceLogger.log_step(
            step_id="deduplication",
            path="(global)",
            status="Deduplicated",
            summary=f"Reduced results from {original_count} to {len(state.responses)}",
            state=state
        )

        # 🔄 Restore relaxed constraints (if applicable)
        if state.relaxation_log and getattr(state, "last_dropped_constraints", []):
            for c in state.last_dropped_constraints:
                ids = data_registry.get(c.key, {}).get(str(c.value), set())
                if ids and constraint_tree:
                    constraint_tree.constraints.append(c)

        # 🧠 Evaluate constraint tree and inject validation
        if constraint_tree and not getattr(state, "constraint_tree_evaluated", False):
            ids = evaluate_constraint_tree(constraint_tree, data_registry)
            if ids:
                inject_validation_steps_from_ids(ids, state)
            state.constraint_tree_evaluated = True

        # 🖼️ Format result
        renderer = RESPONSE_RENDERERS.get(
            state.response_format or "summary", format_fallback)
        state.formatted_response = renderer(state)

        # 🧠 Final explanation
        explanation_lines = []
        if getattr(state, "satisfied_roles", None):
            explanation_lines.append(
                f"✅ Roles satisfied via intersection: {', '.join(sorted(state.satisfied_roles))}."
            )
        explanation_lines.append(QueryExplanationBuilder.build_final_explanation(
            extraction_result=state.extraction_result,
            relaxed_parameters=state.relaxed_parameters,
            fallback_used=any(step.get("fallback_injected")
                              for step in state.plan_steps)
        ))
        state.explanation = "\n".join(explanation_lines)

        if getattr(state, "relaxed_parameters", []):
            relaxed_summary = ", ".join(sorted(set(state.relaxed_parameters)))
            ExecutionTraceLogger.log_step(
                "relaxation_summary",
                path="(global)",
                status="Relaxation Summary",
                summary=f"Relaxed constraints attempted before fallback: {relaxed_summary}",
                state=state
            )

        # 📣 Output final results
        for r in state.responses:
            title = r.get("title") or r.get("name") or "<untitled>"
                f"🧠 Final Result: {title} — score: {r.get('final_score')} — constraints: {r.get('_provenance', {}).get('matched_constraints', [])}"
            )

        log_summary(state)
        return state
