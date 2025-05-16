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

        # 🔁 Symbol-free queries bypass filtering
        if is_symbol_free_query(state):
            for movie in results:
                movie["_step"] = step
                movie["type"] = "movie_summary"
                movie["final_score"] = movie.get("vote_average", 0) / 10.0
                state.responses.append(movie)

            ExecutionTraceLogger.log_step(
                step_id, endpoint, "Handled (symbol-free)",
                summary=f"{len(results)} result(s) extracted (no post-validation)",
                state=state
            )
            return

        # ✅ Only apply validation if endpoint supports symbolic filtering
        validated = results
        if is_symbolically_filterable(endpoint):
            validated = DiscoveryHandler._run_post_validations(
                step, json_data, state)

        if not validated:
            # 🔁 Try relaxing
            relaxed_steps = FallbackHandler.relax_constraints(
                step, state=state)
            if relaxed_steps:
                for retry_step in relaxed_steps:
                    state.plan_steps.insert(0, retry_step)
                state.completed_steps.append(step_id)
                return

            # 🔧 Inject fallback
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

        # 🧠 Rank and filter
        query_entities = state.extraction_result.get("query_entities", [])
        ranked = EntityAwareReranker.boost_by_entity_mentions(
            validated, query_entities)

        for movie in ranked:
            movie["type"] = "movie_summary"
            movie["_step"] = step
            score = movie.get("final_score", 0)
            matched = movie.get("_provenance", {}).get(
                "matched_constraints", [])

            print(
                f"🎯 SCORE: {movie.get('title')} — {score} — constraints matched: {matched}")

            if score > 0 and matched:
                state.responses.append(movie)
                print(
                    f"✅ [APPENDED] {movie.get('title')} — score={score} — matched: {matched}")
            else:
                print(
                    f"❌ [REJECTED] {movie.get('title')} — score={score} — matched: {matched}")

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
                    constraint_tree=state.constraint_tree,
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

            try:
                enrich_symbolic_registry(
                    summary,
                    state.data_registry,
                    credits=None,
                    keywords=None,
                    release_info=None,
                    watch_providers=None
                )
            except Exception as e:
                print(
                    f"⚠️ Failed enrichment for fallback result {summary.get('title') or summary.get('name')}: {e}")

            # Embed metadata on each result
            for summary in summaries:
                summary["source"] = path
                summary["_step"] = step
                summary["type"] = "tv_summary" if "tv" in path else "movie_summary"
                summary["final_score"] = summary.get("final_score", 1.0)

                # ✅ Enrich fallback summary
                from core.entity.param_utils import enrich_symbolic_registry
                try:
                    enrich_symbolic_registry(
                        summary,
                        state.data_registry,
                        credits=None,
                        keywords=None,
                        release_info=None,
                        watch_providers=None
                    )
                except Exception as e:
                    print(
                        f"⚠️ Failed enrichment for fallback result {summary.get('title') or summary.get('name')}: {e}")

            resolved_path = PathRewriter.rewrite(path, state.resolved_entities)

            # 🔁 If symbol-free, skip constraint filtering entirely
            if is_symbol_free_query(state) or not is_symbolically_filterable(resolved_path):
                print(f"✅ Skipping symbolic filtering for: {resolved_path}")
                filtered = filter_symbolic_responses(state, summaries, path)
                state.responses.extend(filtered)
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


def filter_symbolic_responses(state, summaries, endpoint):
    """
    Filters a list of summary dicts based on:
    - final_score > 0
    - symbolic constraint satisfaction (if required for endpoint)
    """
    filtered = []
    for s in summaries:
        score = s.get("final_score", 0)
        if score <= 0:
            continue

        if should_apply_symbolic_filter(state, {"endpoint": endpoint}):
            if passes_symbolic_filter(s, state.constraint_tree, state.data_registry):
                filtered.append(s)
        else:
            filtered.append(s)

    return filtered
