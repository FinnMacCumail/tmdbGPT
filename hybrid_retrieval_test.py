import json
import os
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -- Init clients
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("tmdb_endpoints")

# -- Load test queries
with open("data/sample-tmdb-questions.markdown", "r") as f:
    queries = [q.strip() for q in f if q.strip()]

def extract_intent_entities(query):
    prompt = f"""
    Extract intents and entities from the user's query using this schema:

    {{
        "intents": ["recommendation.similarity", "recommendation.suggested", "discovery.filtered",
                    "discovery.genre_based", "discovery.temporal", "discovery.advanced",
                    "search.basic", "search.multi", "media_assets.image", "media_assets.video",
                    "details.movie", "details.tv", "credits.movie", "credits.tv", "credits.person",
                    "trending.popular", "trending.top_rated", "reviews.movie", "reviews.tv",
                    "collections.movie", "companies.studio", "companies.network"],
        "entities": ["movie", "tv", "person", "company", "network", "collection", "genre", "year", "keyword", "credit", "rating", "date"],
        "query_entities": ["names, titles or specific things directly mentioned by the user"]
    }}

    User Query: \"{query}\"
    Respond with ONLY valid JSON. No commentary:
    """

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Extract intent and entity data as JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("❌ Invalid JSON:\n", content)
        return None

def score_match(user_extraction, candidate_metadata):
    user_intents = set(user_extraction.get("intents", []))
    user_entities = set(user_extraction.get("entities", []))
    query_entities = set(user_extraction.get("query_entities", []))

    try:
        endpoint_intents = json.loads(candidate_metadata.get("intents", "[]"))
    except json.JSONDecodeError:
        endpoint_intents = []

    if isinstance(endpoint_intents, dict):
        endpoint_intents = [endpoint_intents]

    endpoint_entities = set(candidate_metadata.get("entities", "").split(", "))
    path = candidate_metadata.get("path", "")

    # --- Normalized intent score
    matched_intents = [ei for ei in endpoint_intents if ei.get("intent") in user_intents]
    intent_score = sum(float(i.get("confidence", 0)) for i in matched_intents)
    if matched_intents:
        intent_score /= len(matched_intents)
    intent_score = min(intent_score, 1.0)

    # --- Weighted entity score
    weights = {"movie": 0.5, "year": 0.3, "genre": 0.4, "rating": 0.4, "person": 0.4, "date": 0.3}
    entity_score = sum(weights.get(e, 0.1) for e in user_entities & endpoint_entities)

    # --- Parameter compatibility / overlap bonus
    param_boost = len(user_entities & endpoint_entities) / max(1, len(user_entities))

    # --- Boost for entrypoint paths
    if any(e in user_entities for e in {"person", "movie", "tv"}) and "search" in path:
        entity_score += 0.5

    # --- Boost discover/movie if genre + rating present
    if "/discover/movie" in path and {"genre", "rating"}.issubset(user_entities):
        entity_score += 0.3

    # --- Entity mismatch penalty
    generic = {"keyword", "date", "rating", "country"}
    unrelated_entities = endpoint_entities - user_entities - generic
    mismatch_penalty = 0.15 * len(unrelated_entities)

    # --- Penalize if movie query returns TV results
    if "movie" in user_entities and "tv" in endpoint_entities:
        mismatch_penalty += 0.15

    # --- Penalize search endpoints for trending intent
    if any(i.startswith("trending") for i in user_intents) and "/search" in path:
        mismatch_penalty += 0.2

    score = intent_score + entity_score + 0.5 * param_boost - mismatch_penalty
    return round(score, 3)

def semantic_retrieval(extraction_result):
    embedding_text = json.dumps(extraction_result)
    query_embedding = embedder.encode(embedding_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["metadatas", "distances"]
    )

    matches = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        score = score_match(extraction_result, metadata)
        final_score = 0.7 * score + 0.3 * (1 - distance)
        matches.append({
            "endpoint": metadata["path"],
            "intents": metadata.get("intents", ""),
            "entities": metadata.get("entities", ""),
            "distance": round(distance, 4),
            "match_score": round(score, 3),
            "final_score": round(final_score, 3)
        })

    matches.sort(key=lambda x: x["final_score"], reverse=True)
    return matches

def main():
    all_results = []
    print(f"📦 Running {len(queries)} queries from sample-tmdb-questions...\n")

    for user_query in queries:
        extraction = extract_intent_entities(user_query)
        if not extraction:
            print(f"⚠️ Skipped due to extraction failure: {user_query}")
            continue

        matches = semantic_retrieval(extraction)

        result_block = {
            "query": user_query,
            "extraction": extraction,
            "results": matches
        }

        all_results.append(result_block)

        print(f"🧠 Query: {user_query}")
        for m in matches[:3]:
            print(f"  - {m['endpoint']} (score: {m['match_score']}, dist: {m['distance']}, final: {m['final_score']})")

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/retrieval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Retrieval log written to: {log_path}")

if __name__ == "__main__":
    main()
