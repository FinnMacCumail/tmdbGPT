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
        "entities": ["movie", "tv", "person", "company", "network", "collection", "genre", "year", "keyword", "credit"],
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
    """
    Hybrid match score: weighted sum of
    - intent overlap (confidence-based)
    - entity overlap
    """

    user_intents = set(user_extraction.get("intents", []))
    user_entities = set(user_extraction.get("entities", []))

    # --- Parse candidate intents (JSON string of intent+confidence dicts)
    try:
        endpoint_intents = json.loads(candidate_metadata.get("intents", "[]"))
    except json.JSONDecodeError:
        endpoint_intents = []

    if isinstance(endpoint_intents, dict):
        endpoint_intents = [endpoint_intents]

    endpoint_entities = set(candidate_metadata.get("entities", "").split(", "))

    # --- Intent confidence sum
    intent_score = 0.0
    for user_intent in user_intents:
        for ei in endpoint_intents:
            if isinstance(ei, dict) and ei.get("intent") == user_intent:
                intent_score += float(ei.get("confidence", 0))

    # --- Entity intersection
    entity_score = len(user_entities & endpoint_entities) * 0.2

    return round(intent_score + entity_score, 2)

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
        matches.append({
            "endpoint": metadata["path"],
            "intents": metadata.get("intents", ""),
            "entities": metadata.get("entities", ""),
            "distance": round(distance, 4),
            "match_score": score
        })

    matches.sort(key=lambda x: (x["match_score"], -x["distance"]), reverse=True)
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
            print(f"  - {m['endpoint']} (score: {m['match_score']}, dist: {m['distance']})")

    # Save output log
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/retrieval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Retrieval log written to: {log_path}")

if __name__ == "__main__":
    main()
