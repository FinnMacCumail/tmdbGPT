# test_end_to_end_reranking.py

from app import AppState, plan
from nlp_retriever import RerankPlanning
from plan_validator import PlanValidator

def test_end_to_end_reranking():

    # --- 1. Simulate a fake AppState ---
    fake_state = AppState()
    fake_state.input = "Show me best movies released after 2015 with high ratings"
    fake_state.retrieved_matches = [
        {
            "endpoint": "/discover/movie",
            "supported_parameters": [
                "vote_average.gte", "primary_release_year", "release_date.gte", "with_genres"
            ],
            "semantic_score": 0.85,
            "final_score": 0.85
        },
        {
            "endpoint": "/movie/popular",
            "supported_parameters": [],
            "semantic_score": 0.83,
            "final_score": 0.83
        },
        {
            "endpoint": "/trending/movie/week",
            "supported_parameters": ["time_window", "media_type"],
            "semantic_score": 0.81,
            "final_score": 0.81
        }
    ]
    fake_state.resolved_entities = {}

    # --- 2. Inject the query into resolved_entities ---
    fake_state.resolved_entities["__query"] = fake_state.input

    # --- 3. Run plan() ---
    state_after_plan = plan(fake_state)

    # --- 4. Print the reranked results ---
    for step in state_after_plan.execution_steps:
        pass  # Debug output removed

if __name__ == "__main__":
    test_end_to_end_reranking()
