from execution_orchestrator import ExecutionOrchestrator
from dependency_manager import DependencyManager
from hybrid_retrieval_test import semantic_retrieval, convert_matches_to_execution_steps
from llm_client import OpenAILLMClient
from fallback_handler import FallbackHandler
from entity_resolution import TMDBEntityResolver
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import time
from nlp_retriever import RerankPlanning, ResponseFormatter

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {os.getenv('TMDB_API_KEY')}"}

openai_client = OpenAILLMClient()
dependency_manager = DependencyManager()
orchestrator = ExecutionOrchestrator(BASE_URL, HEADERS)
entity_resolver = TMDBEntityResolver(os.getenv('TMDB_API_KEY'), HEADERS)

class AppState(BaseModel):
    input: str
    status: Optional[str] = None
    step: Optional[str] = None
    extraction_result: Optional[Dict] = Field(default_factory=dict)
    resolved_entities: Optional[Dict] = Field(default_factory=dict)
    retrieved_matches: Optional[List] = Field(default_factory=list)
    plan_steps: Optional[List] = Field(default_factory=list)
    responses: Optional[List] = Field(default_factory=list)
    error: Optional[str] = None  # allows setting error message
    data_registry: Optional[Dict] = Field(default_factory=dict)  # for orchestrator context
    completed_steps: Optional[List[str]] = Field(default_factory=list)
    pending_steps: Optional[List[Dict]] = Field(default_factory=list)

def parse(state: AppState) -> AppState:
    print("→ running node: PARSE")
    return state.model_copy(update={"status": "parsed", "step": "parse", "__write_guard__": f"parse_{int(time.time()*1000)}"})

def extract_entities(state: AppState) -> AppState:
    print("→ running node: EXTRACT_ENTITIES")    
    extraction = openai_client.extract_entities_and_intents(state.input)
    print("📤 Extracted entities:")
    for k, v in extraction.items():
        print(f" - {k}: {v}")
    if not extraction:
        return state.model_copy(update={"extraction_result": {}, "step": "extract_entities_failed"})
    return state.model_copy(update={"extraction_result": extraction, "step": "extract_entities_ok"})

def resolve_entities(state: AppState) -> AppState:
    print("→ running node: RESOLVE_ENTITIES")
    extraction = state.extraction_result
    query_entities = extraction.get("query_entities", [])
    base_entities = set(extraction.get("entities", []))  # from LLM extraction

    # ✅ Use multi-entity resolver
    resolved_entities, unresolved_entities = entity_resolver.resolve_entities(query_entities)

    # TMDB-style format: {type_id: [ids]}
    resolved_by_type = {}
    for ent in resolved_entities:
        # Determine the actual resolved type (e.g. fallback to 'company' instead of 'network')
        resolved_type = ent.get("resolved_type", ent["type"])
        resolved_id = ent["resolved_id"]
        key = f"{resolved_type}_id"

        # Update the resolved_entities map
        resolved_by_type.setdefault(key, []).append(resolved_id)

        # Print resolution status with optional fallback annotation
        fallback_tag = " (fallback)" if ent.get("resolved_as") == "fallback" else ""
        print(f"✅ Resolved '{ent['name']}' as {resolved_type} → {resolved_id}{fallback_tag}")


    if not resolved_by_type:
        print("⚠️ No query_entities could be resolved.")

    # Ensure resolved types are reflected in entity list
    for ent in resolved_entities:
        if ent["type"] not in extraction["entities"]:
            extraction["entities"].append(ent["type"])

    return state.model_copy(update={
        "resolved_entities": resolved_by_type,
        "extraction_result": extraction,
        "step": "resolve_entities"
    })


def retrieve_context(state: AppState) -> AppState:
    print("→ running node: RETRIEVE_CONTEXT")
    retrieved_matches = semantic_retrieval(state.extraction_result)
    return state.model_copy(update={"retrieved_matches": retrieved_matches, "step": "retrieve_context"})

def plan(state: AppState) -> AppState:
    print("→ running node: PLAN")

    # Phase 1: Rerank semantic matches using resolved entities
    ranked_matches = RerankPlanning.rerank_matches(state.retrieved_matches, state.resolved_entities)

    # Phase 2: Filter to executable steps
    feasible, deferred = RerankPlanning.filter_feasible_steps(ranked_matches, state.resolved_entities)

    # Phase 3: Convert to execution-ready step format
    execution_steps = convert_matches_to_execution_steps(
        feasible, 
        state.extraction_result, 
        state.resolved_entities
    )

    # Phase 4: Deduplicate steps based on endpoint + parameter signature
    seen = set()
    deduped_steps = []

    for step in execution_steps:
        sig = (step["endpoint"], frozenset(step.get("parameters", {}).items()))
        if sig not in seen:
            seen.add(sig)
            deduped_steps.append(step)
        else:
            print(f"🔁 Skipping duplicate step: {step['endpoint']} with same parameters")

    # Phase 5: Filter out low-signal noisy loops
    signal_steps = []
    for step in deduped_steps:
        endpoint = step["endpoint"]
        params = step.get("parameters", {})

        if "with_people" in params and not (
            "/discover/" in endpoint or "/search/" in endpoint or "/person/" in endpoint
        ):
            print(f"🧹 Removed low-signal step: {endpoint} with with_people")
            continue

        signal_steps.append(step)

    combined_steps = signal_steps

    # Phase 6: Show final execution plan or fallback
    print("\n🧭 Final Execution Plan:")
    for s in combined_steps:
        print(f"→ {s['endpoint']} with params: {s.get('parameters', {})}")

    if not combined_steps:
        print("⚠️ No executable steps found. Using fallback...")
        combined_steps = FallbackHandler.generate_steps(state.resolved_entities, state.extraction_result)

    return state.model_copy(update={
        "plan_steps": combined_steps,
        "step": "plan"
    })

def execute(state: AppState) -> AppState:
    print("→ running node: EXECUTE")
    dependency_manager.analyze_dependencies(state.plan_steps)
    updated_state = orchestrator.execute(state.model_copy(update={"pending_steps": state.plan_steps}))
    return updated_state.model_copy(update={"executed": True, "step": "execute"})

def respond(state: AppState) -> AppState:
    print("→ running node: RESPOND")

    # 1. Collect validated titles from /discover/movie summaries
    validated_titles = {
        r["title"].strip().lower()
        for r in state.responses
        if isinstance(r, dict)
        and r.get("source") == "/discover/movie"
        and r.get("type") == "movie_summary"
    }

    # 2. Filter down string-based fallbacks if they duplicate validated results
    filtered = []
    last_title = None
    for r in state.responses:
        if isinstance(r, str):
            if r.startswith("🎬"):
                last_title = r[2:].split(":")[0].strip().lower()
            if r.startswith("📦 Source: /person") and last_title in validated_titles:
                print(f"🧹 Suppressing fallback for: {last_title}")
                continue
        filtered.append(r)

    state.responses = filtered

    output = ResponseFormatter.format_responses(state.responses)
    if not output:
        output = ["⚠️ No results matched all filters."]
    return state.model_copy(update={"status": "done", "step": "respond", "responses": output})

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
    print("\n--- STARTING GRAPH ---")
    graph = build_app_graph()

    while True:
        user_input = input("\nAsk something (or type 'exit' to quit): ")
        if user_input.lower() in {"exit", "quit"}:
            break
        print("Initial input state:", {"input": user_input})
        result = graph.invoke({"input": user_input})
        print("\n--- RESPONSE ---")
        print(result["responses"])
