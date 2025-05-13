from core.planner.entity_reranker import RoleAwareReranker
from core.execution.execution_orchestrator import ExecutionOrchestrator
from core.planner.dependency_manager import DependencyManager
from core.planner.plan_utils import route_symbol_free_intent
from core.embeddings.hybrid_retrieval import rank_and_score_matches, convert_matches_to_execution_steps
from core.llm.extractor import extract_entities_and_intents
from core.execution.fallback import FallbackHandler
from core.entity.entity_resolution import TMDBEntityResolver
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from nlp.nlp_retriever import RerankPlanning
from core.planner.plan_validator import SymbolicConstraintFilter, PlanValidator
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
    return state.model_copy(update={"status": "parsed", "step": "parse", "__write_guard__": f"parse_{int(time.time()*1000)}"})


def extract_entities(state: AppState) -> AppState:
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


def resolve_entities(state: AppState) -> AppState:
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


def retrieve_context(state: AppState) -> AppState:
    retrieved_matches = rank_and_score_matches(state.extraction_result)
    return state.model_copy(update={"retrieved_matches": retrieved_matches, "step": "retrieve_context"})


def plan(state: AppState) -> AppState:

    if not state.extraction_result.get("query_entities") and not state.extraction_result.get("entities"):
        print("⚠️ No query entities or constraints — using symbol-free planner override")
        return route_symbol_free_intent(state)

    builder = ConstraintBuilder()
    state.constraint_tree = builder.build_from_query_entities(
        state.extraction_result.get("query_entities", [])
    )

    if "input" in state.__dict__:
        state.resolved_entities["__query"] = state.input

    ranked_matches = RerankPlanning.rerank_matches(
        state.retrieved_matches, state.resolved_entities
    )
    ranked_matches = SymbolicConstraintFilter.apply(
        ranked_matches, extraction_result=state.extraction_result,
        resolved_entities=state.resolved_entities
    )
    RoleAwareReranker.boost_matches_by_role(
        ranked_matches, extraction_result=state.extraction_result,
        intended_media_type=state.intended_media_type
    )
    ranked_matches = sorted(
        ranked_matches,
        key=lambda m: m.get("_boost_score", 0) - m.get("_demote_score", 0),
        reverse=True
    )

    feasible, deferred = RerankPlanning.filter_feasible_steps(
        ranked_matches, state.resolved_entities,
        extraction_result=state.extraction_result
    )

    intended_type = state.intended_media_type
    if intended_type:
        feasible = [
            step for step in feasible
            if intended_type == "both"
            or step.get("endpoint", "").startswith(f"/discover/{intended_type}")
            or step.get("endpoint", "").startswith("/person/")
        ]

    execution_steps = convert_matches_to_execution_steps(
        feasible, state.extraction_result, state.resolved_entities
    )

    optional_params = PlanValidator().infer_semantic_parameters(state.input)
    for step in execution_steps:
        step.setdefault("parameters", {})
        for param_name in optional_params:
            if param_name in SAFE_OPTIONAL_PARAMS and param_name not in step["parameters"]:
                step["parameters"][param_name] = "<dynamic_value_or_prompt>"

    dependency_steps = DependencyManager.expand_plan_with_dependencies(
        state, state.resolved_entities)
    if dependency_steps:
        execution_steps.extend(dependency_steps)

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

    if not signal_steps:
        signal_steps = FallbackHandler.generate_steps(
            state.resolved_entities, state.extraction_result)

    return state.model_copy(update={"plan_steps": signal_steps, "step": "plan"})


def execute(state: AppState) -> AppState:
    dependency_manager.analyze_dependencies(state)
    updated_state = orchestrator.execute(state.model_copy(update={
        "pending_steps": state.plan_steps,
        "question_type": state.extraction_result.get("question_type"),
        "response_format": state.extraction_result.get("response_format"),
        "base_url": BASE_URL,
        "headers": HEADERS
    }))
    return updated_state.model_copy(update={"plan_steps": []})


def respond(state: AppState):
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
    graph = build_app_graph()
    while True:
        user_input = input("\nAsk something (or type 'exit' to quit): ")
        if user_input.lower() in {"exit", "quit"}:
            break
        result = graph.invoke({"input": user_input})
        print("\n--- RESPONSE ---")
        print(result["responses"])
