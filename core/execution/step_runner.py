# core/execution/step_runner.py

from typing import List, Dict, Any
from .trace_logger import ExecutionTraceLogger
from .discovery_handler import DiscoveryHandler
from .post_execution_validator import PostExecutionValidator
from .fallback import FallbackPlanner
from planner.plan_validator import PlanValidator
from nlp import PathRewriter, PostStepUpdater
from core.formatting.registry import RESPONSE_RENDERERS
from core.formatting.templates import format_fallback
from core.constraint_model import evaluate_constraint_tree
from hashlib import sha256
import requests
from core.execution.discovery_handler import DiscoveryHandler


class StepRunner:
    def __init__(self, base_url: str, headers: Dict[str, str]):
        self.base_url = base_url
        self.headers = headers
        self.validator = PlanValidator()

    def execute(self, state):
        seen_step_keys = set()
        step_origin_depth = {}
        MAX_CHAIN_DEPTH = 3

        if not hasattr(state, "roles_injected"):
            state.roles_injected = False

        while state.plan_steps:
            step = state.plan_steps.pop(0)
            step_id = step.get("step_id")
            endpoint = step.get("endpoint")
            params = step.get("parameters", {})

            # Skip if already completed
            if step_id in state.completed_steps:
                continue

            # Inject path slots
            step = self.validator.inject_path_slot_parameters(
                step,
                resolved_entities=state.resolved_entities,
                extraction_result=state.extraction_result
            )

            # Resolve path
            path = PathRewriter.rewrite(endpoint, state.resolved_entities)
            full_url = f"{self.base_url}{path}"

            # Dedup check
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
                    state.data_registry[step_id] = json_data

                    # Post-process
                    state = PostStepUpdater.update(state, step, json_data)

                    # Handle discovery or fallback
                    if "/discover/movie" in endpoint or "/discover/tv" in endpoint:
                        DiscoveryHandler.handle_discover_step(
                            step, json_data, state)
                    else:
                        DiscoveryHandler.handle_result_step(
                            step, json_data, state)

                    if not state.roles_injected:
                        expected_role_steps = {
                            f"step_{qe['role']}_{qe['resolved_id']}"
                            for qe in state.extraction_result.get("query_entities", [])
                            if qe.get("type") == "person" and qe.get("role")
                        }

                        if expected_role_steps.issubset(set(state.completed_steps)):
                            print(
                                "✅ All role steps complete → injecting lookup steps via role intersection")
                            from core.planner.dependency_manager import (
                                analyze_dependencies,
                                inject_lookup_steps_from_role_intersection
                            )
                            state = analyze_dependencies(state)
                            state = inject_lookup_steps_from_role_intersection(
                                state)
                            state.roles_injected = True  # 🛡 prevent duplicate injection
                else:
                    ExecutionTraceLogger.log_step(
                        step_id, path, f"HTTP Error {response.status_code}", state=state)
            except Exception as ex:
                ExecutionTraceLogger.log_step(
                    step_id, path, f"Failed: {ex}", state=state)
                state.error = str(ex)

            state.completed_steps.append(step_id)

        # Format final response
        renderer = RESPONSE_RENDERERS.get(
            state.response_format or "summary", format_fallback)
        state.formatted_response = renderer(state)

        return state
