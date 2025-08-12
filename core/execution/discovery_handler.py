# core/execution/discovery_handler.py

from core.execution.trace_logger import ExecutionTraceLogger

from core.planner.entity_reranker import EntityAwareReranker
from core.constraint_model import evaluate_constraint_tree
from core.execution.fallback import FallbackHandler, FallbackSemanticBuilder
from core.entity.param_utils import update_symbolic_registry

from core.execution.post_execution_validator import PostExecutionValidator
from nlp.nlp_retriever import ResultExtractor
from core.entity.param_utils import enrich_symbolic_registry
from core.entity.symbolic_filter import passes_symbolic_filter, lazy_enrich_and_filter

from core.planner.plan_validator import should_apply_symbolic_filter
from core.planner.plan_utils import is_symbol_free_query, filter_valid_movies_or_tv

from nlp.nlp_retriever import PathRewriter
from core.planner.plan_utils import is_symbolically_filterable
import requests
from core.model.constraint import Constraint
from core.planner.constraint_planner import inject_validation_steps_from_ids
from core.planner.plan_utils import extract_matched_constraints
from core.execution.post_validator import PostValidator
from core.validation.role_validators import validate_roles
from core.execution.financial_filter import FinancialFilter


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

            # 🚩 Mark endpoint as handled to suppress fallbacks for fact queries
            if summaries:
                handled = state.extraction_result.setdefault(
                    "__handled_endpoints", [])
                if path not in handled:
                    handled.append(path)

            if not summaries:
                return

            resolved_path = PathRewriter.rewrite(path, state.resolved_entities)

            # 🧩 Attach safe metadata
            for summary in summaries:
                summary["source"] = path
                summary["_step"] = step
                # 🔎 Preserve existing type if set (e.g., person_profile)
                if "type" not in summary:
                    if "/movie/" in path and path.count("/") == 2:
                        summary["type"] = "movie_summary"
                    elif "/tv/" in path and path.count("/") == 2:
                        summary["type"] = "tv_summary"
                    elif "/person/" in path and path.count("/") == 2:
                        summary["type"] = "person_profile"
                    elif "/company/" in path:
                        summary["type"] = "company_profile"
                    elif "/network/" in path:
                        summary["type"] = "network_profile"
                    elif "/collection/" in path:
                        summary["type"] = "collection_profile"
                    else:
                        summary["type"] = "generic_summary"
                summary["final_score"] = summary.get("final_score", 1.0)

            # 🧩 Only enrich movie/TV summaries
            enrichable = [
                s for s in summaries
                if s.get("type") in ("movie_summary", "tv_summary")
            ]

            # 🔁 Enrich TV/movie summaries with symbolic keys (e.g. cast_4495)
            for summary in enrichable:
                enrich_symbolic_registry(
                    movie=summary,
                    registry=state.data_registry,
                    credits=None,  # Already extracted → we only need _actor_id
                    keywords=None,
                    release_info=None,
                    watch_providers=None
                )

                # 🎬 Attach director(s) if available in json_data["credits"]
                if summary.get("type") == "movie_summary":
                    credits = json_data.get("credits", {})
                    directors = [
                        p.get("name") for p in credits.get("crew", [])
                        if p.get("job", "").lower() == "director" and p.get("name")
                    ]
                    if directors:
                        summary["directors"] = directors

            # 🔁 Skip symbolic filtering if not needed
            if is_symbol_free_query(state) or not is_symbolically_filterable(resolved_path):
                # Debug output removed
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
                            # Debug output removed
                            filtered.append(summary)
                        else:
                            # Debug output removed
                            pass
                except Exception as e:
                    # Debug output removed
                    pass

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
    def handle_discover_step(step, step_id, path, json_data, state, depth, seen_step_keys):

        # Determine media type from endpoint
        media_type = "tv" if "/tv" in step.get("endpoint", "") else "movie"

        # 🎯 Post-validate discovery results using symbolic constraints (e.g., cast, director, genre).
        # If validation rules are triggered (e.g., with_people), performs credit lookups for accuracy.
        filtered = PostValidator.run_post_validations(step, json_data, state)

        # 💰 Apply financial filtering if revenue constraints are present
        if filtered and FinancialFilter.should_apply_financial_filtering(state.constraint_tree):
            result_limit = FinancialFilter.estimate_result_limit(state.constraint_tree)
            
            # 💡 Fetch revenue data for each movie before applying revenue filters
            # /discover/movie doesn't return revenue data, so we need individual lookups
            for movie in filtered:
                movie_id = movie.get('id')
                if movie_id and 'revenue' not in movie:
                    try:
                        detail_url = f"{state.base_url}/movie/{movie_id}"
                        detail_response = requests.get(detail_url, headers=state.headers)
                        if detail_response.status_code == 200:
                            detail_data = detail_response.json()
                            movie['revenue'] = detail_data.get('revenue', 0)
                    except Exception:
                        # If we can't fetch details, set to 0 (will be filtered out)
                        movie['revenue'] = 0
            
            filtered = FinancialFilter.apply_financial_filters(
                filtered, state.constraint_tree, max_results=result_limit
            )

        # ❌ No results passed validation or symbolic filtering.
        # Triggers fallback or relaxation logic (e.g., retry with fewer constraints or inject generic discovery).
        if not filtered:
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
            filtered, query_entities)

        # 🧩 Inject symbolic enrichment before constraint filtering
        for item in ranked:
            enrich_symbolic_registry(
                movie=item,
                registry=state.data_registry,
                credits=None,
                keywords=None,
                release_info=None,
                watch_providers=None
            )

        # 🧠 Apply Symbolic Filtering (if supported)
        # If the current step supports symbolic constraint filtering (e.g., /discover/movie),
        # filter results using the constraint tree and symbolic registry to enforce role, genre, and entity alignment.
        # Otherwise, skip filtering and accept all ranked results as valid (e.g., for detail or fallback steps).
        if should_apply_symbolic_filter(state, step):
            valid_mt = filter_valid_movies_or_tv(
                ranked,
                constraint_tree=state.constraint_tree,
                registry=state.data_registry
            )
        else:
            valid_mt = ranked  # skip filtering

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
        for item in valid_mt:
            item_id = item.get("id")
            item["_step"] = step  # 🔧 Embed step metadata
            if not item_id:
                continue

            # Enrich with TMDB /credits if available
            credits = fetch_credits_for_entity(
                item, state.base_url, state.headers, state)

            # ✅ Inject provenance and enrich registry
            item["final_score"] = item.get("final_score", 1.0)
            item["type"] = "movie_summary"
            item["_provenance"] = {
                "matched_constraints": matched,
                "relaxed_constraints": relaxed,
                "post_validations": validated
            }

            if credits:
                validate_roles(
                    credits=credits,
                    query_entities=query_entities,
                    movie=item,
                    state=state
                )

                enrich_symbolic_registry(
                    item,
                    state.data_registry,
                    credits=credits,
                    keywords=None,
                    release_info=None,
                    watch_providers=None
                )
            else:
                # Debug output removed
                item.setdefault("_provenance", {}).setdefault(
                    "post_validations", []).append("missing_credits")

            satisfied = item["_provenance"].get("satisfied_roles", [])
            state.satisfied_roles.update(satisfied)

            # Debug output removed
            state.responses.append(item)

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
    
    Special handling for person credit endpoints:
    - Single-person queries bypass constraint validation
    - Multi-person queries maintain validation
    """
    filtered = []
    
    # Check if this is a person credit endpoint requiring special handling
    is_person_credits = '/person/' in endpoint and '/movie_credits' in endpoint
    
    if is_person_credits:
        # Analyze constraint tree to determine query complexity
        constraints = getattr(state.constraint_tree, 'constraints', []) if hasattr(state, 'constraint_tree') else []
        person_constraints = [c for c in constraints if hasattr(c, 'key') and c.key == 'with_people']
        
        # Single-person query detection - only bypass for purely person-based queries
        is_single_person = (len(person_constraints) == 1 and len(constraints) == 1)
        
        if is_single_person:
            # For single-person queries from person credits, bypass constraint validation
            # These summaries are already the correct result set from the person endpoint
            # Timeline queries like "First movies by Steven Spielberg" need this bypass
            for s in summaries:
                if s.get("final_score", 0) > 0:
                    filtered.append(s)
            return filtered
    
    # Standard filtering logic for all other cases
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


def fetch_credits_for_entity(entity: dict, base_url: str, headers: dict, state=None) -> dict:
    """
    Fetch and return credits for either a movie or TV entity, with caching.

    Args:
        entity (dict): Entity object with "id" and optionally "media_type"
        base_url (str): TMDB base URL
        headers (dict): HTTP headers
        state (AppState, optional): Used for caching credits

    Returns:
        dict: A TMDB credits object with cast and crew fields, or empty dict on failure
    """
    entity_id = entity.get("id")
    if not entity_id:
        return {}

    # Detect type (fallback to 'movie' if not specified)
    media_type = entity.get("type", entity.get("media_type", "movie"))
    if media_type not in ("movie", "tv"):
        media_type = "movie"

    # Init credits cache
    if state is not None and "credits_cache" not in state.data_registry:
        state.data_registry["credits_cache"] = {}

    cache_key = f"{media_type}_{entity_id}"
    if state and cache_key in state.data_registry["credits_cache"]:
        return state.data_registry["credits_cache"][cache_key]

    try:
        url = f"{base_url}/{media_type}/{entity_id}/credits"
        # Debug output removed
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if state:
                state.data_registry["credits_cache"][cache_key] = data
            return data
        else:
            # Debug output removed
            pass
    except Exception as e:
        # Debug output removed
        pass

    return {}
