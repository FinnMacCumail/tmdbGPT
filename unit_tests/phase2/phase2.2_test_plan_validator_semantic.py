# test_plan_validator_semantic.py

from plan_validator import PlanValidator

def test_infer_semantic_parameters():
    validator = PlanValidator()

    test_queries = [
        "Find the best rated movies",
        "Show me recent action movies",
        "List TV shows released after 2020",
        "Find highly voted thrillers",
        "What are long runtime epic films?",
    ]

    for query in test_queries:
        suggested_params = validator.infer_semantic_parameters(query)

if __name__ == "__main__":
    test_infer_semantic_parameters()
