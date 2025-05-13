# core/execution/discovery_handler.py

from core.execution.trace_logger import ExecutionTraceLogger

from core.planner.entity_reranker import EntityAwareReranker
from core.constraint_model import evaluate_constraint_tree
from core.execution.fallback import FallbackHandler, FallbackSemanticBuilder
from core.entity.param_utils import update_symbolic_registry

from core.execution.post_execution_validator import PostExecutionValidator
from core.execution.post_validator import PostValidator
from nlp.nlp_retriever import ResultExtractor
from core.entity.param_utils import enrich_symbolic_registry
from core.entity.symbolic_filter import passes_symbolic_filter

from core.planner.plan_validator import should_apply_symbolic_filter
from core.planner.plan_utils import is_symbol_free_query

from nlp.nlp_retriever import PathRewriter
from core.planner.plan_utils import is_symbolically_filterable


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

        # 🔁 Symbol-free: skip validation/fallback
        if is_symbol_free_query(state):
            for movie in results:
                movie["_step"] = step  # ✅ always embed step
                movie["type"] = "movie_summary"
                movie["final_score"] = movie.get("vote_average", 0) / 10.0
                state.responses.append(movie)

            ExecutionTraceLogger.log_step(
                step_id, endpoint, "Handled (symbol-free)",
                summary=f"{len(results)} result(s) extracted (no post-validation)",
                state=state
            )
            return

        # 🔍 Step 1: Validate and score
        validated = DiscoveryHandler._run_post_validations(
            step, json_data, state)
        if not validated:
            # 🔁 Try relaxing constraints
            relaxed_steps = FallbackHandler.relax_constraints(
                step, state=state)
            if relaxed_steps:
                for retry_step in relaxed_steps:
                    state.plan_steps.insert(0, retry_step)
                state.completed_steps.append(step_id)
                return

            # 🔧 Inject semantic fallback
            fallback_step = FallbackSemanticBuilder.enrich_fallback_step(
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

        # 🧠 Step 2: Rank and append
        query_entities = state.extraction_result.get("query_entities", [])
        ranked = EntityAwareReranker.boost_by_entity_mentions(
            validated, query_entities)

        print(f"📊 Ranked results count: {len(ranked)}")

        for movie in ranked:
            movie["type"] = "movie_summary"
            movie["final_score"] = movie.get("final_score", 1.0)
            movie["_step"] = step  # ✅ always embed step context

            enrich_symbolic_registry(movie, state.data_registry)

            step_context = movie["_step"]
            title = movie.get("title") or movie.get("name", "Untitled")

            # ✅ For non-filterable endpoints, accept as-is
            if not should_apply_symbolic_filter(state, step_context):
                print(
                    f"✅ [NO FILTER] {title} → appended without symbolic filtering")
                state.responses.append(movie)
                continue

            # ✅ Score movie against all constraints
            score, matched_constraints = PostValidator.score_movie_against_query(
                movie,
                state.constraint_tree,
                state.data_registry
            )
            movie["score"] = score
            movie["_provenance"] = {"satisfied_roles": matched_constraints}

            if score > 0:
                print(
                    f"✅ [MATCHED] {title} → score: {score} | roles: {matched_constraints}")
                state.responses.append(movie)
            else:
                print(
                    f"❌ [REJECTED] {title} → score: 0 | roles: {matched_constraints}")

        # 🗃️ Store validated entries
        state.data_registry[step_id]["validated"] = ranked
        ExecutionTraceLogger.log_step(
            step_id, endpoint, "Validated", summary=ranked[0], state=state
        )

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
        Applies symbolic filtering only when necessary.
        """
        path = step["endpoint"]
        try:
            summaries = ResultExtractor.extract(
                json_data, path, state.resolved_entities)

            if not summaries:
                return

            # Embed metadata on each result
            for summary in summaries:
                summary["source"] = path
                summary["_step"] = step
                summary["type"] = "tv_summary" if "tv" in path else "movie_summary"
                summary["final_score"] = summary.get("final_score", 1.0)

            resolved_path = PathRewriter.rewrite(path, state.resolved_entities)

            # 🔁 If symbol-free, skip constraint filtering entirely
            if is_symbol_free_query(state) or not is_symbolically_filterable(resolved_path):
                print(f"✅ Skipping symbolic filtering for: {resolved_path}")
                state.responses.extend(summaries)
                ExecutionTraceLogger.log_step(
                    step["step_id"], path,
                    "Handled (non-filtered endpoint)",
                    summary=f"{len(summaries)} result(s) used directly without filtering",
                    state=state
                )
                return

            # 🧪 Apply symbolic constraint filtering
            filtered = [
                m for m in summaries
                if passes_symbolic_filter(m, state.constraint_tree, state.data_registry)
            ]

            state.responses.extend(filtered)
            ExecutionTraceLogger.log_step(
                step["step_id"], path,
                "Handled (filtered)",
                summary=f"{len(filtered)} of {len(summaries)} result(s) kept after symbolic filtering",
                state=state
            )

        except Exception as e:
            ExecutionTraceLogger.log_step(
                step["step_id"], path,
                status=f"Failed during generic extract: {e}",
                state=state
            )
