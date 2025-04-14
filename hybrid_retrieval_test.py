import json
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb
from llm_client import OpenAILLMClient
from entity_reranker import EntityAwareReranker


load_dotenv()

# Init clients
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("tmdb_endpoints")
openai_client = OpenAILLMClient()

def hybrid_search(prompt: str, top_k: int = 10) -> list:
    """
    Interpret a natural language prompt to generate structured extraction,
    then perform hybrid semantic retrieval.

    Args:
        prompt (str): user-style query like "Fetch endpoints requiring person_id for intent credits.person"
        top_k (int): number of results to return

    Returns:
        list: retrieved endpoint matches
    """
    structured = openai_client.extract_entities_and_intents(prompt)
    return semantic_retrieval(structured, top_k=top_k)

def extract_intent_entities(openai, query):
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

    # --- Dynamic entity weighting
    weights = {"movie": 0.5, "year": 0.3, "genre": 0.4, "rating": 0.4, "person": 0.4, "date": 0.3}
    if "discovery.filtered" in user_intents:
        weights.update({"rating": 0.6, "year": 0.5})
    elif "trending.popular" in user_intents:
        weights.update({"year": 0.1, "rating": 0.1})

    entity_score = sum(weights.get(e, 0.1) for e in user_entities & endpoint_entities)

    # --- Parameter compatibility score
    try:
        endpoint_params = json.loads(candidate_metadata.get("parameters", "[]"))
        param_names = {p.get("name", "") for p in endpoint_params if isinstance(p, dict)}
    except Exception:
        param_names = set()
    param_overlap = len([e for e in user_entities if e in param_names])
    param_boost = param_overlap / max(1, len(user_entities))

    # --- Boost for entrypoint paths
    if any(e in user_entities for e in {"person", "movie", "tv"}) and "search" in path:
        entity_score += 0.5

    # --- Boost discover/movie if genre + rating present
    if "/discover/movie" in path and {"genre", "rating"}.issubset(user_entities):
        entity_score += 0.3
    
    # --- Boost discover/movie if genre + rating + year all present
    if "/discover/movie" in path and {"genre", "rating", "year"}.issubset(user_entities):
        entity_score += 0.4

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

    # --- Penalize general intent unless no other match
    if "general" in [i.get("intent") for i in endpoint_intents]:
        mismatch_penalty += 0.2

    score = intent_score + entity_score + 0.5 * param_boost - mismatch_penalty
    return round(score, 3)

def semantic_retrieval(extraction_result, top_k=10):
    embedding_text = json.dumps(extraction_result)
    query_embedding = embedder.encode(embedding_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
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

    matches = EntityAwareReranker.boost_by_entity_mentions(matches, extraction_result.get("query_entities", []))
    return matches

def convert_matches_to_execution_steps(matches, extraction_result, resolved_entities):
    steps = []
    query_entity = (extraction_result.get("query_entities") or [None])[0]

    for i, match in enumerate(matches):
        endpoint = match["endpoint"]
        parameters = match.get("parameters", {}).copy()

        # Inject resolved entities into path/query parameters
        for entity_key, entity_value in resolved_entities.items():
            if entity_key in endpoint:
                parameters[entity_key] = entity_value

        # Special handling: inject "query" into /search/* endpoints
        if "/search/" in endpoint and "query" not in parameters and query_entity:
            parameters["query"] = query_entity

        step_id = f"step_{i:06x}"
        step = {
            "step_id": step_id,
            "endpoint": endpoint,
            "parameters": parameters
        }
        steps.append(step)

    return steps
