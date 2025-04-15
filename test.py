from hybrid_retrieval_test import hybrid_search

query = "Find movies with multiple actors like Robert De Niro and Al Pacino"
results = hybrid_search(query, top_k=10)
for r in results:
    print(r.get("endpoint") or r.get("path"), r.get("score"))