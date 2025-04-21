import json
import os
from tqdm import tqdm
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

PARAM_TO_ENTITY_MAP = {
    "first_air_date_year": "date",
    "region": "country",
    "year": "date",
    "primary_release_year": "date",
    "certification_country": "country",
    "primary_release_date.gte": "date",
    "primary_release_date.lte": "date",
    "release_date.gte": "date",
    "release_date.lte": "date",
    "vote_average.gte": "rating",
    "vote_average.lte": "rating",
    "with_people": "person",
    "with_companies": "company",
    "with_genres": "genre",
    "without_genres": "genre",
    "with_keywords": "keyword",
    "without_keywords": "keyword",
    "with_original_language": "language",
    "with_watch_providers": "watch_provider",
    "watch_region": "country",
    "air_date.gte": "date",
    "air_date.lte": "date",
    "first_air_date.gte": "date",
    "first_air_date.lte": "date",
    "with_networks": "network",
    "movie_id": "movie",
    "person_id": "person",
    "tv_id": "tv",
    "network_id": "network",
    "company_id": "company",
    "collection_id": "collection"
}

client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("tmdb_endpoints")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/tmdb.json") as f:
    raw = json.load(f)
    index = raw.get("paths", {})

def _detect_intents(path, description):
    intents = []
    path_lower = path.lower()
    desc = description.lower()
    if "/person/" in path_lower and "/movie_credits" in path_lower:
        intents += ["credits.person"]
    if "/person/" in path_lower and "/tv_credits" in path_lower:
        intents += ["credits.person"]
    if "/person/" in path_lower and "/images" in path_lower:
        intents.append("media_assets.image")
    if "/movie/" in path_lower and "/keywords" in path_lower:
        intents.append("keywords.movie")
    if "/tv/" in path_lower and "/keywords" in path_lower:
        intents.append("keywords.tv")
    if "/movie/" in path_lower and "/similar" in path_lower:
        intents.append("recommendation.similarity")
    if "/movie/" in path_lower and "/recommendations" in path_lower:
        intents.append("recommendation.suggested")
    if "/movie/" in path_lower:
        intents.append("details.movie")
    if "/tv/" in path_lower:
        intents.append("details.tv")
    if "/collection/" in path_lower:
        intents.append("collection.details")
    if "/company/" in path_lower:
        intents += ["company.details", "companies.studio"]
    if "/network/" in path_lower:
        intents += ["network.details", "companies.network"]
    if "/person/" in path_lower:
        intents += ["details.person"]
    if "/genre/movie" in path_lower:
        intents.append("genre.list.movie")
    if "/genre/tv" in path_lower:
        intents.append("genre.list.tv")
    if "/review" in path_lower:
        intents += ["review.lookup", "reviews.movie", "reviews.tv"]
    if "/discover/movie" in path_lower:
        intents += ["discovery.filtered", "discovery.advanced", "discovery.genre_based", "discovery.temporal"]
    if "/discover/tv" in path_lower:
        intents += ["discovery.filtered", "discovery.advanced", "discovery.genre_based", "discovery.temporal"]
    if "/trending" in path_lower:
        intents += ["trending.popular", "trending.top_rated"]
    if "/search/movie" in path_lower:
        intents.append("search.movie")
    if "/search/tv" in path_lower:
        intents.append("search.tv")
    if "/search/person" in path_lower:
        intents.append("search.person")
    if "/search/collection" in path_lower:
        intents.append("search.collection")
    if "/search/company" in path_lower:
        intents.append("search.company")
    if not intents:
        intents.append("miscellaneous")
    return intents

def _detect_entities(endpoint, parameters):
    entities = set()
    for param in parameters:
        entity = PARAM_TO_ENTITY_MAP.get(param)
        if entity:
            entities.add(entity)
    for slot in ["movie_id", "person_id", "tv_id", "company_id", "collection_id", "network_id"]:
        if f"{{{slot}}}" in endpoint:
            entities.add(PARAM_TO_ENTITY_MAP.get(slot))
    return list(entities)

def _detect_produced_entities(endpoint: str, parameters: list) -> list:
    produces = []
    path = endpoint.lower()
    if "/discover/movie" in path:
        produces.append("movie")
    if "/discover/tv" in path:
        produces.append("tv")
    if "/person/" in path and "/movie_credits" in path:
        produces.append("movie")
    if "/person/" in path and "/tv_credits" in path:
        produces.append("tv")
    if "/search/person" in path:
        produces.append("person")
    if "/search/movie" in path:
        produces.append("movie")
    if "/search/tv" in path:
        produces.append("tv")
    if "/search/collection" in path:
        produces.append("collection")
    if "/search/company" in path:
        produces.append("company")
    if "/search/network" in path:
        produces.append("network")
    if "/search/keyword" in path:
        produces.append("keyword")
    if "/movie/" in path and ("/similar" in path or "/recommendations" in path):
        produces.append("movie")
    if "/tv/" in path and ("/similar" in path or "/recommendations" in path):
        produces.append("tv")
    if "/collection/" in path:
        produces.append("movie")
    return list(set(produces))

def _override_description(endpoint_path: str, original: str) -> str:
    if endpoint_path == "/discover/movie":
        return (
            "Endpoint supports discovery.filtered intent. "
            "Entities: person, genre, date. "
            "Parameters: with_people, with_genres, primary_release_year. "
            "Use to find movies filtered by cast, genre, and year. "
            "Example: Find movies with both Robert De Niro and Al Pacino."
        )
    if endpoint_path == "/discover/tv":
        return (
            "Endpoint supports discovery.filtered intent. "
            "Entities: tv, person, genre, company, network. "
            "Parameters: with_people, with_genres, with_networks, with_companies, first_air_date_year. "
            "Use to find TV shows filtered by cast, genre, studio, or year. "
            "Example: Best Netflix crime shows from 2020."
        )
    return original

def _create_embedding_text(path, description):
    return f"{path}\n{_override_description(path, description)}"

def _create_metadata(endpoint_path, obj):
    description = obj.get("description", "")
    parameters = obj.get("parameters", {}).keys()
    return {
        "path": endpoint_path,
        "description": description,
        "intents": json.dumps([{ "intent": i } for i in _detect_intents(endpoint_path, description)]),
        "media_type": "tv" if "/tv" in endpoint_path else "movie" if "/movie" in endpoint_path else "any",
        "consumes_entities": json.dumps(_detect_entities(endpoint_path, parameters)),
        "produces_entities": json.dumps(_detect_produced_entities(endpoint_path, parameters))
    }

def process_endpoints():

    ids, docs, metas, embs = [], [], [], []

    for path, obj in tqdm(index.items()):
        embedding_text = _create_embedding_text(path, obj.get("description", ""))
        metadata = _create_metadata(path, obj)
        embedding = embedder.encode(embedding_text).tolist()

        ids.append(path)
        docs.append(embedding_text)
        embs.append(embedding)
        metas.append(metadata)
    
    print(f"📦 Upserting {len(ids)} endpoints into ChromaDB (tmdb_endpoints)...")
    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=embs,
        metadatas=metas
    )
    print("✅ Done embedding TMDB endpoints")

if __name__ == "__main__":
    process_endpoints()
