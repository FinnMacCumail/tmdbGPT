from hybrid_retrieval_test import hybrid_search

result = hybrid_search("Find movies featuring both Robert De Niro and Al Pacino", top_k=10)

for r in result:
    print(f"{r['endpoint']} → score: {r['final_score']} | entities: {r['entities']}")