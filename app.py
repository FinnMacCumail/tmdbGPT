# Debug mode control - Set to True for development debugging
DEBUG_MODE = False

from core.planner.entity_reranker import RoleAwareReranker
from core.execution.execution_orchestrator import ExecutionOrchestrator
from core.planner.dependency_manager import DependencyManager
from core.planner.plan_utils import route_symbol_free_intent, is_symbol_free_query
from core.embeddings.hybrid_retrieval import rank_and_score_matches, convert_matches_to_execution_steps
from core.llm.extractor import extract_entities_and_intents
from core.execution.fallback import FallbackHandler
from core.entity.entity_resolution import TMDBEntityResolver
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from nlp.nlp_retriever import RerankPlanning
from core.planner.plan_validator import SymbolicConstraintFilter, PlanValidator
from core.model.constraint import ConstraintGroup
from core.formatting.formatter import ResponseFormatter
from core.formatting.templates import format_fallback, format_ranked_list, generate_relaxation_explanation
from core.constraint_model import ConstraintBuilder
from core.execution_state import AppState
import os
import time

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {os.getenv('TMDB_API_KEY')}"}

dependency_manager = DependencyManager()
orchestrator = ExecutionOrchestrator(BASE_URL, HEADERS)
entity_resolver = TMDBEntityResolver(os.getenv('TMDB_API_KEY'), HEADERS)

SAFE_OPTIONAL_PARAMS = {
    "vote_average.gte", "vote_count.gte", "primary_release_year",
    "release_date.gte", "with_runtime.gte", "with_runtime.lte",
    "with_original_language", "region"
}


def parse(state: AppState) -> AppState:
    if not DEBUG_MODE:
        print("🎭 Identifying people, movies, and details...", flush=True)
    return state.model_copy(update={"status": "parsed", "step": "parse", "__write_guard__": f"parse_{int(time.time()*1000)}"})

# 🧠 Entity & Intent Extraction Step
# Converts the user's raw query (state.input) into structured semantic fields using LLM or mock data.
# Extracts: intents, query_entities, media_type, question_type, etc.
# Stores results in state.extraction_result for use by planning and execution modules.
# If extraction fails, logs failure but preserves execution flow.


def extract_entities(state: AppState) -> AppState:
    if not DEBUG_MODE:
        print("🔎 Looking up information...", flush=True)
    
    if hasattr(state, "mock_extraction") and state.mock_extraction:
        extraction = state.mock_extraction
    else:
        extraction = extract_entities_and_intents(state.input)

    if not extraction:
        return state.model_copy(update={"extraction_result": {}, "step": "extract_entities_failed"})
    return state.model_copy(update={
        "extraction_result": extraction,
        "step": "extract_entities_ok",
        "intended_media_type": extraction.get("media_type")
    })

# 🔍 Entity Resolution Step
# Resolves symbolic query entities (e.g., "Netflix", "Brad Pitt") into TMDB IDs using TMDBEntityResolver.
# Outputs:
#   - resolved_entities: dict mapping types to TMDB IDs (e.g., {"person_id": [6193]})
#   - Updates extraction_result with any new entity types (e.g., "company", "network")
# This enables path slot filling and parameter injection in downstream planning.


def resolve_entities(state: AppState) -> AppState:
    if not DEBUG_MODE:
        print("📚 Gathering context...", flush=True)
    
    extraction = state.extraction_result
    query_entities = extraction.get("query_entities", [])
    resolved_entities, unresolved_entities = entity_resolver.resolve_entities(
        query_entities,
        intended_media_type=state.intended_media_type
    )

    resolved_by_type = {}
    for ent in resolved_entities:
        resolved_type = ent.get("resolved_type", ent["type"])
        resolved_id = ent["resolved_id"]
        key = f"{resolved_type}_id"
        resolved_by_type.setdefault(key, []).append(resolved_id)

    for ent in resolved_entities:
        if ent["type"] not in extraction["entities"]:
            extraction["entities"].append(ent["type"])

    return state.model_copy(update={
        "resolved_entities": resolved_by_type,
        "extraction_result": extraction,
        "step": "resolve_entities"
    })

# 🧠 Semantic Endpoint Retrieval
# Uses hybrid semantic search (e.g., ChromaDB + embeddings) to retrieve TMDB API endpoints
# that match the extracted query intent and entity types.
# Outputs a ranked list of endpoint templates with metadata and scores (retrieved_matches),
# which is later refined by symbolic filtering and reranking.


def retrieve_context(state: AppState) -> AppState:
    if not DEBUG_MODE:
        print("🗓️ Planning search strategy...", flush=True)
    
    retrieved_matches = rank_and_score_matches(state.extraction_result)
    return state.model_copy(update={"retrieved_matches": retrieved_matches, "step": "retrieve_context"})


def plan(state: AppState) -> AppState:
    if not DEBUG_MODE:
        print("🎬 Searching movies and shows...", flush=True)
    
    if is_symbol_free_query(state):
        return route_symbol_free_intent(state)

    builder = ConstraintBuilder()

    if state.extraction_result.get("question_type") != "fact":
        state.constraint_tree = builder.build_from_query_entities(
            state.extraction_result.get("query_entities", [])
        )
    else:
        state.constraint_tree = ConstraintGroup([], logic="AND")

    # 🧠 Inject raw query into resolved_entities for downstream access.
    # Some validators (e.g., semantic parameter inference) need access to the original query string.
    # Storing it under '__query' ensures it's available during reranking or enrichment phases.
    if "input" in state.__dict__:
        state.resolved_entities["__query"] = state.input

    # 📊 Symbolic-Aware Reranking
    # Reorders retrieved endpoint candidates based on their ability to fulfill the query.
    # Applies penalties for missing required entities (e.g., person_id),
    # and boosts for supporting optional parameters inferred from the raw query (e.g., year, rating).
    # Outputs a sorted list of endpoint templates prioritized by final_score
    ranked_matches = RerankPlanning.rerank_matches(
        state.retrieved_matches, state.resolved_entities
    )

    # 🧠 Apply Symbolic Constraint Penalties
    # Evaluates symbolic feasibility of each ranked match based on the query's resolved entities.
    # Penalizes endpoints that lack support for required symbolic parameters (e.g., with_people, with_genres).
    # Adds `.penalty` fields to influence downstream final_score calculation.
    ranked_matches = SymbolicConstraintFilter.apply(
        ranked_matches, extraction_result=state.extraction_result,
        resolved_entities=state.resolved_entities
    )

    # 🎯 Role-Aware Boosting
    # Boosts endpoint matches that align with person roles extracted from the query (e.g., actor, director).
    # Applies additional score if the endpoint meaningfully supports that role (e.g., /movie_credits for directors).
    # Helps prioritize role-specific endpoints over generic ones.
    RoleAwareReranker.boost_matches_by_role(
        ranked_matches, extraction_result=state.extraction_result,
        intended_media_type=state.intended_media_type
    )

    # 📊 Final Sort by Boost-Demote Delta
    # Sorts the reranked endpoint matches using the net effect of role-aware and symbolic scoring:
    #   effective_score = _boost_score - _demote_score
    # This prioritizes symbolically aligned endpoints over semantically similar but incompatible ones.
    # Note: final_score is preserved but not used here — this is a symbolic sensitivity adjustment.
    ranked_matches = sorted(
        ranked_matches,
        key=lambda m: m.get("_boost_score", 0) - m.get("_demote_score", 0),
        reverse=True
    )

    # ✅ Filter Feasible vs. Deferred Execution Steps
    # Scans all ranked endpoint matches and filters them into:
    # - feasible: endpoints with all required path parameters resolved (e.g., person_id, genre_id)
    # - deferred: endpoints that are valid but currently missing critical parameters
    # Special overrides apply for role-aware 'count' queries (e.g., allow director credit counts)
    feasible, deferred = RerankPlanning.filter_feasible_steps(
        ranked_matches, state.resolved_entities,
        extraction_result=state.extraction_result
    )

    # 🎯 Filter Feasible Steps by Intended Media Type
    # Ensures only endpoints matching the intended media type (tv/movie) are kept.
    # - Keeps /discover/{media_type} steps matching the query (e.g., /discover/tv)
    # - Also allows /person/{person_id}/... role-based steps (media-neutral)
    # - Skips incompatible /discover endpoints (e.g., movie steps in a TV query)
    intended_type = state.intended_media_type
    if intended_type:
        feasible = [
            step for step in feasible
            if intended_type == "both"
            or step.get("endpoint", "").startswith(f"/discover/{intended_type}")
            or step.get("endpoint", "").startswith("/person/")
        ]

    # 🔧 Convert Feasible Matches into Executable API Steps
    # Transforms each feasible endpoint template into a structured execution step:
    # - Injects resolved parameters (e.g., with_people, with_genres)
    # - Assigns step_id, endpoint, method, and metadata
    # - Prepares for actual TMDB API calls in the execution engine
    execution_steps = convert_matches_to_execution_steps(
        feasible, state.extraction_result, state.resolved_entities
    )

    # 🎯 Inject Optional Semantic Parameters
    # Uses semantic inference to guess helpful filter parameters based on the query text
    # (e.g., rating, year, language) and adds them to each step if:
    # - They're in the SAFE_OPTIONAL_PARAMS list
    # - They’re not already set in the step
    # Inserts a placeholder "<dynamic_value_or_prompt>" for downstream filling or prompting
    optional_params = PlanValidator().infer_semantic_parameters(state.input)
    for step in execution_steps:
        step.setdefault("parameters", {})
        for param_name in optional_params:
            if param_name in SAFE_OPTIONAL_PARAMS and param_name not in step["parameters"]:
                step["parameters"][param_name] = "<dynamic_value_or_prompt>"

    # 🔗 Expand Plan with Dependency-Based Steps
    # Adds symbolic follow-up steps based on newly resolved entities:
    # - Role-aware credit lookups for each person (e.g., director → /movie_credits)
    # - Collection and TV lookups
    # - Fallback discovery steps for companies or networks (e.g., with_networks → /discover/tv)
    # Enables multi-hop planning and chaining across related endpoints.
    dependency_steps = DependencyManager.expand_plan_with_dependencies(
        state, state.resolved_entities)

    # ➕ Append Dependency-Based Steps
    if dependency_steps:
        execution_steps.extend(dependency_steps)

    # Ensures final step list is both unique and symbolically valid
    # Filters out invalid uses of symbolic people filters (e.g., with_people on non-discovery endpoints)
    seen = set()
    deduped_steps = []
    for step in execution_steps:
        sig = (step["endpoint"], frozenset(step.get("parameters", {}).items()))
        if sig not in seen:
            seen.add(sig)
            deduped_steps.append(step)

    signal_steps = []
    for step in deduped_steps:
        endpoint = step["endpoint"]
        params = step.get("parameters", {})
        if "with_people" in params and not any(k in endpoint for k in ["/discover/", "/search/", "/person/"]):
            continue
        signal_steps.append(step)

    # If all planned steps have been filtered out or deduplicated away,
    # fall back to a generic discovery/search plan using resolved entities and high-level intent.
    # Prevents empty responses and ensures execution can still proceed.
    if not signal_steps:
        signal_steps = FallbackHandler.generate_steps(
            resolved_entities=state.resolved_entities,
            intents=state.extraction_result,
            extraction_result=state.extraction_result)

    return state.model_copy(update={"plan_steps": signal_steps, "step": "plan"})


def execute(state: AppState) -> AppState:
    if not DEBUG_MODE:
        print("✨ Preparing your results...", flush=True)
    
    updated_state = orchestrator.execute(state.model_copy(update={
        "pending_steps": state.plan_steps,
        "question_type": state.extraction_result.get("question_type"),
        "response_format": state.extraction_result.get("response_format"),
        "base_url": BASE_URL,
        "headers": HEADERS
    }))
    return updated_state.model_copy(update={"plan_steps": []})


def respond(state: AppState):
    if not DEBUG_MODE:
        print("📋 Formatting your results...", flush=True)
    
    if state.formatted_response:
        return {"responses": state.formatted_response}

    if state.response_format == "ranked_list":
        lines = format_ranked_list(state, include_debug=True)
    else:
        lines = ResponseFormatter.format_responses(state)

    if not lines:
        fallback = format_fallback(state)
        lines = fallback.get("entries", ["⚠️ No explanation available."])

    if state.relaxed_parameters:
        relax_expl = generate_relaxation_explanation(state.relaxed_parameters)
        lines.insert(0, relax_expl)

    return {"responses": lines}


def build_app_graph():
    builder = StateGraph(AppState)
    builder.add_node("parse", parse)
    builder.add_node("extract_entities", extract_entities)
    builder.add_node("resolve_entities", resolve_entities)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("plan", plan)
    builder.add_node("execute", execute)
    builder.add_node("respond", respond)

    builder.set_entry_point("parse")
    builder.add_edge("parse", "extract_entities")
    builder.add_edge("extract_entities", "resolve_entities")
    builder.add_edge("resolve_entities", "retrieve_context")
    builder.add_edge("retrieve_context", "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "respond")
    builder.set_finish_point("respond")

    return builder.compile()


if __name__ == "__main__":
    # Initialize application
    graph = build_app_graph()
    
    while True:
        user_input = input("\nAsk something (or type 'exit' to quit): ")
        
        if user_input.lower() in {"exit", "quit"}:
            break
            
        # Show user-friendly progress indicators (unless in debug mode)
        if not DEBUG_MODE:
            print("🔍 Understanding your question...", flush=True)
        
        # Process user query through the application graph
        result = graph.invoke({"input": user_input})
        
        # Display the final formatted response to user
        if result and "formatted_response" in result:
            print("\n" + "="*60)
            print(result["formatted_response"])
            print("="*60)
        elif result and "responses" in result and result["responses"]:
            print("\n" + "="*60)
            for response_line in result["responses"]:
                print(response_line)
            print("="*60)
        else:
            print("\nNo results found.")
