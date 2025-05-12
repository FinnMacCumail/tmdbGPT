# core/execution/discovery_handler.py

from core.execution.trace_logger import ExecutionTraceLogger

from core.planner.entity_reranker import EntityAwareReranker
from core.constraint_model import evaluate_constraint_tree
from core.execution.fallback import FallbackHandler
from core.entity.param_utils import update_symbolic_registry

from core.execution.post_execution_validator import PostExecutionValidator
from core.execution.post_validator import PostValidator
from nlp.nlp_retriever import ResultExtractor
from core.entity.param_utils import enrich_symbolic_registry
from core.entity.symbolic_filter import passes_symbolic_filter


class DiscoveryHandler:

    @staticmethod
    def handle_discover_step(step, json_data, state):
        step_id = step.get("step_id")
        endpoint = step.get("endpoint")
        results = json_data.get("results", [])

        if not results:
            ExecutionTraceLogger.log_step(
                step_id, endpoint, "Empty", state=state)
            return

        # Step 1: Validate and score
        validated = DiscoveryHandler._run_post_validations(
            step, json_data, state)
        if not validated:
            # No valid results → try relaxing
            relaxed_steps = FallbackHandler.relax_constraints(
                step, state=state)
            if relaxed_steps:
                for retry_step in relaxed_steps:
                    state.plan_steps.insert(0, retry_step)
                state.completed_steps.append(step_id)
                return

            # Last resort: fallback
            fallback_step = FallbackHandler.enrich_fallback_step(
                original_step=step,
                extraction_result=state.extraction_result,
                resolved_entities=state.resolved_entities
            )
            if fallback_step["step_id"] not in state.completed_steps:
                state.plan_steps.insert(0, fallback_step)
                ExecutionTraceLogger.log_step(
                    fallback_step["step_id"],
                    fallback_step["endpoint"],
                    status="Semantic Fallback Injected",
                    summary=f"Injected fallback step with parameters: {fallback_step['parameters']}",
                    state=state
                )
            state.completed_steps.append(step_id)
            return

        # Step 2: Rank and append results
        query_entities = state.extraction_result.get("query_entities", [])
        ranked = EntityAwareReranker.boost_by_entity_mentions(
            validated, query_entities)

        for movie in ranked:
            movie["type"] = "movie_summary"
            movie["final_score"] = movie.get("final_score", 1.0)
            enrich_symbolic_registry(
                movie, state.data_registry, credits=credits)
            if passes_symbolic_filter(movie, state.constraint_tree, state.data_registry):
                state.responses.append(movie)

        state.data_registry[step_id]["validated"] = ranked
        ExecutionTraceLogger.log_step(
            step_id, endpoint, "Validated", summary=ranked[0], state=state)

    @staticmethod
    def _run_post_validations(step, json_data, state):
        """
        Validate movie results against symbolic constraint tree and role presence.
        """
        results = json_data.get("results", [])
        validated = []

        for movie in results:
            movie_id = movie.get("id")
            if not movie_id:
                continue

            credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
            try:
                import requests
                res = requests.get(credits_url, headers=state.headers)
                if res.status_code != 200:
                    continue
                credits = res.json()
            except Exception:
                continue

            score, matched_constraints = PostValidator.score_movie_against_query(
                movie=movie,
                state=state,
                credits=credits,
                step=step,
                query_entities=state.extraction_result.get(
                    "query_entities", [])
            )

            if score > 0:
                movie["final_score"] = min(score, 1.0)
                movie["_provenance"] = movie.get("_provenance", {})
                movie["_provenance"]["matched_constraints"] = matched_constraints
                validated.append(movie)

        return validated

    @staticmethod
    def handle_generic_response(step, json_data, state):
        """
        Extract generic summaries from non-discovery endpoints and attach to state.
        """

        path = step["endpoint"]
        try:
            summaries = ResultExtractor.extract(
                json_data, path, state.resolved_entities)
            if summaries:
                for summary in summaries:
                    summary["source"] = path
                filtered = [
                    m for m in summaries
                    if passes_symbolic_filter(m, state.constraint_tree, state.data_registry)
                ]
                state.responses.extend(filtered)

                ExecutionTraceLogger.log_step(
                    step["step_id"], path,
                    "Handled",
                    summary=f"{len(summaries)} result(s) extracted.",
                    state=state
                )
        except Exception as e:
            ExecutionTraceLogger.log_step(
                step["step_id"], path,
                status=f"Failed during generic extract: {e}",
                state=state
            )
