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
from collections import defaultdict
from nlp_retriever import JoinStepExpander

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {os.getenv('TMDB_API_KEY')}"}

openai_client = OpenAILLMClient()
dependency_manager = DependencyManager()
orchestrator = ExecutionOrchestrator(BASE_URL, HEADERS)
entity_resolver = TMDBEntityResolver(os.getenv('TMDB_API_KEY'))

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

def resolve_entities(state):
    extraction = state.extraction_result
    query_entities = extraction.get("query_entities", [])
    resolved = {}

    print("🔍 Resolving typed query_entities from LLM...")
    for item in query_entities:
        if not isinstance(item, dict):
            print(f"⚠️ Skipping unstructured item: {item}")
            continue

        val = item.get("name")
        entity_type = item.get("type")
        if not val or not entity_type:
            print(f"⚠️ Skipping invalid entity object: {item}")
            continue

        match = entity_resolver.resolve_entity(val, entity_type)
        if match:
            print(f"✅ Resolved '{val}' as {entity_type} → {match}")
            key = f"{entity_type}_id"
            if key not in resolved:
                resolved[key] = []
            resolved[key].append(match)

            if entity_type not in extraction["entities"]:
                extraction["entities"].append(entity_type)
                print(f"➕ Added '{entity_type}' to extraction_result['entities']")
        else:
            print(f"❌ Failed to resolve '{val}' as {entity_type}")

    if not resolved:
        print("⚠️ No query_entities could be resolved.")

    return state.model_copy(update={
        "resolved_entities": resolved,
        "extraction_result": extraction,
        "step": "resolve_entities"
    })

def retrieve_context(state: AppState) -> AppState:
    print("→ running node: RETRIEVE_CONTEXT")
    retrieved_matches = semantic_retrieval(state.extraction_result)
    return state.model_copy(update={"retrieved_matches": retrieved_matches, "step": "retrieve_context"})

def plan(state: AppState) -> AppState:
    print("→ running node: PLAN")

    ranked_matches = RerankPlanning.rerank_matches(state.retrieved_matches, state.resolved_entities)
    feasible, deferred = RerankPlanning.filter_feasible_steps(ranked_matches, state.resolved_entities)

    execution_steps = convert_matches_to_execution_steps(feasible, state.extraction_result, state.resolved_entities)

    # Phase 9.2: attempt to enrich plan with join-compatible endpoints
    join_steps = JoinStepExpander.suggest_join_steps(
        resolved_entities=state.resolved_entities,
        extraction_result=state.extraction_result
    )

    combined_steps = execution_steps + join_steps
    print("\n🧭 Final Execution Plan:")
    for s in combined_steps:
        print(f"→ {s['endpoint']} with params: {s.get('parameters', {})}")
    for step in join_steps:
        print(f"🧩 Join step injected: {step['endpoint']} params={step['parameters']}")


    if not combined_steps:
        combined_steps = FallbackHandler.generate_steps(state.resolved_entities, state.extraction_result)

    return state.model_copy(update={"plan_steps": combined_steps, "step": "plan"})

def execute(state: AppState) -> AppState:
    print("→ running node: EXECUTE")
    dependency_manager.analyze_dependencies(state.plan_steps)
    updated_state = orchestrator.execute(state.model_copy(update={"pending_steps": state.plan_steps}))
    return updated_state.model_copy(update={"executed": True, "step": "execute"})

def respond(state: AppState) -> AppState:
    print("→ running node: RESPOND")
    output = ResponseFormatter.format_responses(state.responses)
    if not output:
        output = ["No valid results were returned."]
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
