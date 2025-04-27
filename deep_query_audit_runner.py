import os
from datetime import datetime
from app import build_app_graph
import json

# 🔥 Config
USE_MOCK_EXTRACTION = True
SAVE_AUDIT_REPORT = True
AUDIT_MD_PATH = "audit_summary.md"

SAMPLE_QUERIES = [
    "Find TV shows from Netflix about crime",
    "Best action movies starring Dwayne Johnson",
    "Horror movies produced by Blumhouse",
    "Top rated sci-fi TV shows from Amazon Prime",
    "Comedy movies directed by Taika Waititi",
    "Trending documentaries from the last year",
    "TV shows featuring Zendaya",
    "Best animated movies from Pixar released after 2015",
    "British drama TV series from BBC",
    "Adventure movies starring Harrison Ford",
    "TV shows from HBO Max about politics",
    "Romantic comedies starring Reese Witherspoon",
    "Find movies distributed by A24 about coming of age",
    "Best Korean dramas released after 2020",
    "Mystery TV shows with twists produced by FX",
    "Movies about space travel released after 2010",
    "TV shows directed by Ryan Murphy",
    "Best horror TV shows on Hulu",
    "Best action movies directed by Christopher Nolan",
    "TV series set in medieval times produced by Starz",
    "Top fantasy movies based on books",
    "TV shows featuring Idris Elba in a leading role",
    "Psychological thrillers from the 2000s",
    "Best animated TV series for adults",
    "TV shows about survival on Netflix",
]

# --- Init graph
graph = build_app_graph()

# --- Simple smart mock
def better_mock_extraction(query):
    query_lower = query.lower()
    query_entities = []
    intents = []
    entities = []
    question_type = "list"
    response_format = "ranked_list"

    # --- Base media type
    if "tv" in query_lower:
        intents.append("discovery.filtered.tv")
        entities.append("tv")
    else:
        intents.append("discovery.filtered.movie")
        entities.append("movie")

    # --- Streaming Services (with resolved_id)
    if "netflix" in query_lower:
        query_entities.append({"name": "Netflix", "type": "network", "resolved_id": 213})
    if "amazon prime" in query_lower or "prime video" in query_lower:
        query_entities.append({"name": "Amazon Prime Video", "type": "network", "resolved_id": 1024})
    if "hulu" in query_lower:
        query_entities.append({"name": "Hulu", "type": "network", "resolved_id": 453})
    if "hbo max" in query_lower or "hbo" in query_lower:
        query_entities.append({"name": "HBO", "type": "network", "resolved_id": 49})
    if "starz" in query_lower:
        query_entities.append({"name": "Starz", "type": "network", "resolved_id": 318})
    if "bbc" in query_lower:
        query_entities.append({"name": "BBC One", "type": "network", "resolved_id": 4})
    if "a24" in query_lower:
        query_entities.append({"name": "A24", "type": "company", "resolved_id": 41077})
    if "blumhouse" in query_lower:
        query_entities.append({"name": "Blumhouse Productions", "type": "company", "resolved_id": 3172})
    if "pixar" in query_lower:
        query_entities.append({"name": "Pixar", "type": "company", "resolved_id": 3})

    # --- People (with fake IDs)
    if "dwayne johnson" in query_lower:
        query_entities.append({"name": "Dwayne Johnson", "type": "person", "role": "cast", "resolved_id": 18918})
    if "taika waititi" in query_lower:
        query_entities.append({"name": "Taika Waititi", "type": "person", "role": "director", "resolved_id": 55934})
    if "zendaya" in query_lower:
        query_entities.append({"name": "Zendaya", "type": "person", "role": "cast", "resolved_id": 505710})
    if "reese witherspoon" in query_lower:
        query_entities.append({"name": "Reese Witherspoon", "type": "person", "role": "cast", "resolved_id": 1979})
    if "harrison ford" in query_lower:
        query_entities.append({"name": "Harrison Ford", "type": "person", "role": "cast", "resolved_id": 3})
    if "ryan murphy" in query_lower:
        query_entities.append({"name": "Ryan Murphy", "type": "person", "role": "director", "resolved_id": 1216938})
    if "christopher nolan" in query_lower:
        query_entities.append({"name": "Christopher Nolan", "type": "person", "role": "director", "resolved_id": 525})
    if "idris elba" in query_lower:
        query_entities.append({"name": "Idris Elba", "type": "person", "role": "cast", "resolved_id": 17605})

    # --- Genres (with fake IDs matching TMDB genres)
    if "horror" in query_lower:
        query_entities.append({"name": "Horror", "type": "genre", "resolved_id": 27})
    if "comedy" in query_lower:
        query_entities.append({"name": "Comedy", "type": "genre", "resolved_id": 35})
    if "fantasy" in query_lower:
        query_entities.append({"name": "Fantasy", "type": "genre", "resolved_id": 14})
    if "mystery" in query_lower:
        query_entities.append({"name": "Mystery", "type": "genre", "resolved_id": 9648})
    if "thriller" in query_lower:
        query_entities.append({"name": "Thriller", "type": "genre", "resolved_id": 53})
    if "sci-fi" in query_lower or "science fiction" in query_lower:
        query_entities.append({"name": "Science Fiction", "type": "genre", "resolved_id": 878})
    if "animated" in query_lower or "animation" in query_lower:
        query_entities.append({"name": "Animation", "type": "genre", "resolved_id": 16})
    if "adventure" in query_lower:
        query_entities.append({"name": "Adventure", "type": "genre", "resolved_id": 12})
    if "drama" in query_lower:
        query_entities.append({"name": "Drama", "type": "genre", "resolved_id": 18})
    if "documentary" in query_lower:
        query_entities.append({"name": "Documentary", "type": "genre", "resolved_id": 99})

    # --- Dates (for things like 'after 2015', '2000s')
    if "after 2015" in query_lower:
        query_entities.append({"name": "2015", "type": "date"})
    if "after 2020" in query_lower:
        query_entities.append({"name": "2020", "type": "date"})
    if "2000s" in query_lower:
        query_entities.append({"name": "2000", "type": "date"})
    if "last year" in query_lower:
        from datetime import datetime
        last_year = datetime.now().year - 1
        query_entities.append({"name": str(last_year), "type": "date"})

    return {
        "extraction_result": {
            "query_entities": query_entities,
            "entities": list(set([e["type"] for e in query_entities] + entities)),
            "intents": intents,
            "question_type": question_type,
            "response_format": response_format
        },
        "execution_trace": [],
        "responses": []
    }

# --- Audit Results
audit_results = []
total_queries = len(SAMPLE_QUERIES)
success_count = 0
partial_count = 0
fail_count = 0

print("\n📣 Starting Deep Audit...")

for idx, query in enumerate(SAMPLE_QUERIES, 1):
    print(f"\n▶️ [{idx}/{total_queries}] Query: {query}")
    try:
        if USE_MOCK_EXTRACTION:
            print(f"🤖 MOCK extraction for: {query}")
            mock_extraction = better_mock_extraction(query)["extraction_result"]
            result = graph.invoke({"input": query, "mock_extraction": mock_extraction})
        else:
            result = graph.invoke({"input": query})

        extraction = result.get("extraction_result", {})
        responses = result.get("responses", [])
        explanation = result.get("explanation", "")
        trace = result.get("execution_trace", [])
        relaxed = result.get("relaxed_parameters", [])
        fallback_used = any("fallback" in step.get("step_id", "") for step in trace)

        # --- Scoring
        if not responses or (isinstance(responses, dict) and not responses.get("entries")):
            final_status = "❌ Fail"
            fail_count += 1
        elif relaxed or fallback_used:
            final_status = "⚠️ Partial"
            partial_count += 1
        else:
            final_status = "✅ Success"
            success_count += 1

        audit_results.append({
            "query": query,
            "status": final_status,
            "fallback_used": fallback_used,
            "relaxed_parameters": relaxed,
            "explanation": explanation.strip() if explanation else "No explanation generated.",
            "responses_count": len(responses) if isinstance(responses, list) else len(responses.get("entries", []))
        })

    except Exception as e:
        print(f"❌ Error running query '{query}': {e}")
        audit_results.append({
            "query": query,
            "status": "❌ Fail",
            "fallback_used": False,
            "relaxed_parameters": [],
            "explanation": str(e),
            "responses_count": 0
        })
        fail_count += 1

# --- Save Markdown Report
if SAVE_AUDIT_REPORT:
    with open(AUDIT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(f"# 📋 Deep Query Audit Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"## Overall Stats\n")
        f.write(f"- ✅ Success: {success_count}\n")
        f.write(f"- ⚠️ Partial: {partial_count}\n")
        f.write(f"- ❌ Fail: {fail_count}\n")
        f.write(f"- Total: {total_queries}\n\n")
        f.write("---\n")

        for result in audit_results:
            f.write(f"## Query: {result['query']}\n")
            f.write(f"- **Status:** {result['status']}\n")
            f.write(f"- **Fallback Used:** {result['fallback_used']}\n")
            if result['relaxed_parameters']:
                f.write(f"- **Relaxations:** {', '.join(result['relaxed_parameters'])}\n")
            f.write(f"- **Responses Returned:** {result['responses_count']}\n")
            f.write(f"- **Explanation:** {result['explanation']}\n")
            f.write("\n---\n\n")

    print(f"\n✅ Audit report saved to {AUDIT_MD_PATH}")

# --- Final Report
print("\n📈 Audit Complete:")
print(f"✅ Success: {success_count}")
print(f"⚠️ Partial: {partial_count}")
print(f"❌ Fail: {fail_count}")
