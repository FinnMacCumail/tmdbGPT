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
from core.entity.symbolic_filter import passes_symbolic_filter, lazy_enrich_and_filter, filter_valid_movies

from core.planner.plan_validator import should_apply_symbolic_filter
from core.planner.plan_utils import is_symbol_free_query

from nlp.nlp_retriever import PathRewriter
from core.planner.plan_utils import is_symbolically_filterable
import requests
from core.model.constraint import Constraint
from core.planner.constraint_planner import inject_validation_steps_from_ids


class DiscoveryHandler:

    @staticmethod
    def handle_result_step(step, json_data, state):
        """
        Extract generic summaries from non-discovery endpoints and attach to state.
        Applies symbolic filtering only when necessary, with lazy enrichment.
        """
        path = step["endpoint"]

        try:
            summaries = ResultExtractor.extract(
                json_data, path, state.resolved_entities)
            if not summaries:
                return

            resolved_path = PathRewriter.rewrite(path, state.resolved_entities)

            # 🧩 Attach safe metadata
            for summary in summaries:
                summary["source"] = path
                summary["_step"] = step
                summary["type"] = "tv_summary" if "tv" in path else "movie_summary"
                summary["final_score"] = summary.get("final_score", 1.0)

            # 🔁 Skip symbolic filtering if not needed
            if is_symbol_free_query(state) or not is_symbolically_filterable(resolved_path):
                print(
                    f"✅ Skipping full symbolic filtering for: {resolved_path} — fallback will still use lazy_enrich_and_filter()")
                filtered = filter_symbolic_responses(state, summaries, path)
                state.responses.extend(filtered)
                ExecutionTraceLogger.log_step(
                    step["step_id"], path,
                    "Handled (non-filtered endpoint)",
                    summary=f"{len(filtered)} result(s) used directly without filtering",
                    state=state
                )
                return

            # 🧪 Apply lazy symbolic filtering with fallback enrichment
            filtered = []
            for summary in summaries:
                try:
                    summary["media_type"] = "tv" if "tv" in path else "movie"
                    if lazy_enrich_and_filter(
                        summary,
                        constraint_tree=state.constraint_tree,
                        registry=state.data_registry,
                        headers=state.headers,
                        base_url=state.base_url
                    ):
                        summary["_provenance"] = summary.get("_provenance", {})
                        matched_constraints = extract_matched_constraints(
                            summary, state.constraint_tree, state.data_registry
                        )
                        summary["_provenance"]["matched_constraints"] = matched_constraints

                        if matched_constraints:
                            print(
                                f"✅ [PASSED] {summary.get('title')} — matched: {matched_constraints}")
                            filtered.append(summary)
                        else:
                            print(
                                f"❌ [REJECTED] {summary.get('title')} — matched nothing")
                except Exception as e:
                    print(
                        f"⚠️ Lazy filter failed for {summary.get('title')} → {e}")

            state.responses.extend(filtered)
            ExecutionTraceLogger.log_step(
                step["step_id"], path,
                "Handled (filtered)",
                summary=f"{len(filtered)} of {len(summaries)} result(s) kept after lazy symbolic filtering",
                state=state
            )

        except Exception as e:
            ExecutionTraceLogger.log_step(
                step["step_id"], path,
                status=f"Failed during generic extract: {e}",
                state=state
            )

    @staticmethod
    def handle_discover_movie_step(step, step_id, path, json_data, state, depth, seen_step_keys):
        # 🎯 Post-validate discovery results using symbolic constraints (e.g., cast, director, genre).
        # If validation rules are triggered (e.g., with_people), performs credit lookups for accuracy.
        filtered_movies = PostValidator.run_post_validations(
            step, json_data, state)

        # ❌ No results passed validation or symbolic filtering.
        # Triggers fallback or relaxation logic (e.g., retry with fewer constraints or inject generic discovery).
        if not filtered_movies:
            ExecutionTraceLogger.log_step(
                step_id, path, "Filtered", "No matching results", state=state
            )

            already_dropped = {p.strip()
                               for p in step_id.split("_relaxed_")[1:] if p}

            # 🔁 Trigger Constraint Relaxation
            # If symbolic filtering or validation yielded no results,
            # attempt to recover by relaxing one symbolic constraint (e.g., drop with_genres or with_director).
            # Returns one or more fallback discovery steps to re-attempt execution with looser filters.
            relaxed_steps = FallbackHandler.relax_constraints(
                step, already_dropped, state=state)

            if relaxed_steps:
                for relaxed_step in relaxed_steps:
                    if relaxed_step["step_id"] not in state.completed_steps:
                        constraint_dropped = relaxed_step["step_id"].split("_relaxed_")[
                            1]
                        # ➕ Inject the Relaxed Step at the Front of the Plan
                        state.plan_steps.insert(0, relaxed_step)
                        ExecutionTraceLogger.log_step(
                            relaxed_step["step_id"], path,
                            status=f"Relaxation Injected ({constraint_dropped})",
                            summary=f"Dropped constraint: {constraint_dropped}",
                            state=state
                        )
                # 🧠 Track What’s Been Relaxed
                state.relaxed_parameters.extend(already_dropped)
                ExecutionTraceLogger.log_step(
                    step_id, path, "Relaxation Started", summary="Injected relaxed steps", state=state
                )
                # ✅ Mark the Original Step as Complete
                state.completed_steps.append(step_id)
                return

            # 🧠 Enrich Fallback Discovery Step with Semantic Context
            # When symbolic filtering fails, enhance the original discovery step by injecting optional parameters
            # (e.g., vote_average.gte, primary_release_year, with_original_language) inferred from the query and resolved entities.
            # This produces more relevant fallback results even when symbolic constraints are dropped.
            fallback_step = FallbackSemanticBuilder.enrich_fallback_step(
                original_step=step,
                extraction_result=state.extraction_result,
                resolved_entities=state.resolved_entities
            )
            # 🛠️ Inject Enriched Fallback Step (if not already executed)
            # If a semantic fallback step was generated and hasn't been run yet:
            # - Insert it at the front of the plan to retry with enriched semantic parameters (e.g., rating, year, language)
            # - Log the injection for traceability
            # - Mark the original (failed) step as completed to avoid retrying it
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

        # 🎯 Boost Results Based on Query Entity Matches - intent-aware prioritization
        # Applies a ranking boost to each movie that includes direct matches to the entities mentioned in the user's query
        # (e.g., actor IDs in cast, director IDs in crew, matching production companies).
        # This ensures that results tied more strongly to the user's explicit intent are ranked higher.
        query_entities = state.extraction_result.get("query_entities", [])
        ranked = EntityAwareReranker.boost_by_entity_mentions(
            filtered_movies, query_entities)

        # 🧠 Apply Symbolic Filtering (if supported)
        # If the current step supports symbolic constraint filtering (e.g., /discover/movie),
        # filter results using the constraint tree and symbolic registry to enforce role, genre, and entity alignment.
        # Otherwise, skip filtering and accept all ranked results as valid (e.g., for detail or fallback steps).
        if should_apply_symbolic_filter(state, step):
            valid_movies = filter_valid_movies(
                ranked,
                constraint_tree=state.constraint_tree,
                registry=state.data_registry
            )
        else:
            valid_movies = ranked  # skip filtering

        # 📌 Track matched constraints for explanation
        matched_keys = set(
            e.key for e in state.constraint_tree if isinstance(e, Constraint))
        matched = [f"{c.key}={c.value}" for c in state.constraint_tree if isinstance(
            c, Constraint) and c.key in matched_keys]

        relaxed = list(state.relaxation_log)
        validated = list(state.post_validation_log)

        # 🧠 Initialize role tracker if needed
        if not hasattr(state, "satisfied_roles"):
            state.satisfied_roles = set()

        # 🧪 Process filtered results
        for movie in valid_movies:
            movie_id = movie.get("id")
            movie["_step"] = step  # 🔧 Embed step metadata
            if not movie_id:
                continue

            # Enrich with TMDB /credits if available
            # may nned to cache this as could have already occureed in the pipleline? - refer to How to Avoid Redundant Calls (Optional Optimization)
            credits = fetch_credits_for_entity(
                movie, state.base_url, state.headers)

            # ✅ Inject provenance and enrich registry
            movie["final_score"] = movie.get("final_score", 1.0)
            movie["type"] = "movie_summary"
            movie["_provenance"] = {
                "matched_constraints": matched,
                "relaxed_constraints": relaxed,
                "post_validations": validated
            }

            # needs update
            enrich_symbolic_registry(
                movie,
                state.data_registry,
                credits=credits,
                keywords=None,
                release_info=None,
                watch_providers=None
            )

            satisfied = movie["_provenance"].get("satisfied_roles", [])
            state.satisfied_roles.update(satisfied)

            print(
                f"🧠 Appending validated movie: {movie.get('title')} with score {movie.get('final_score')}")
            state.responses.append(movie)

        state.data_registry[step_id]["validated"] = ranked

        # 🔁 Attempt to Restore Dropped Constraints
        # If symbolic constraints were dropped during fallback, but valid matching results are later found,
        # restore those constraints to the constraint tree.
        # Also update the relaxation log to reflect restored constraints for auditability
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

        # ✅ Check if Evaluation Has Already Occurred
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


def extract_matched_constraints(entity, constraint_tree, registry):
    ids = evaluate_constraint_tree(constraint_tree, registry)
    matched = []
    for constraint in constraint_tree:
        if isinstance(constraint, Constraint):
            cid_sets = ids.get(entity.get("media_type", "movie"), {}).get(
                constraint.key, {})
            value = constraint.value
            # ✅ Normalize list-wrapped values like [4495] → 4495
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if value in cid_sets:
                if entity.get("id") in cid_sets[value]:
                    matched.append(f"{constraint.key}={value}")
    return matched


def fetch_credits_for_entity(entity, base_url, headers):
    """
    Fetch and return credits for either movie or tv.
    """
    media_type = "tv" if entity.get("type", "").startswith("tv") else "movie"
    entity_id = entity.get("id")
    if not entity_id:
        return None

    try:
        url = f"{base_url}/{media_type}/{entity_id}/credits"
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(
            f"⚠️ Could not fetch credits for {media_type} ID {entity_id}: {e}")
    return None
