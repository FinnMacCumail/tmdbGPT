import json
import os
from tqdm import tqdm
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB and SentenceTransformer
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("tmdb_endpoints")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load TMDB endpoints
with open("data/tmdb.json") as f:
    raw = json.load(f)
    index = raw.get("paths", {})

# Load dynamically built parameter-to-entity mapping
with open("data/param_to_entity_map.json") as f:
    PARAM_TO_ENTITY_MAP = json.load(f)


def _detect_intents(path: str, description: str) -> list:
    intents = []
    path_lower = path.lower()
    description_lower = description.lower()

    if "/discover/" in path_lower:
        intents += ["discovery.filtered", "discovery.advanced", "discovery.genre_based", "discovery.temporal"]
    elif "/search/" in path_lower:
        if "/person" in path_lower:
            intents.append("search.person")
        elif "/movie" in path_lower:
            intents.append("search.movie")
        elif "/tv" in path_lower:
            intents.append("search.tv")
        elif "/collection" in path_lower:
            intents.append("search.collection")
        elif "/company" in path_lower:
            intents.append("search.company")
        else:
            intents.append("search.multi")
    elif "/recommendations" in path_lower or "/similar" in path_lower:
        intents.append("recommendation.similarity")
        if "/movie" in path_lower:
            intents.append("recommendation.suggested")
    elif "credits" in path_lower:
        if "/person" in path_lower:
            intents.append("credits.person")
        elif "/movie" in path_lower:
            intents.append("credits.movie")
        elif "/tv" in path_lower:
            intents.append("credits.tv")
        return intents

    elif "/images" in path_lower or "/videos" in path_lower:
        intents.append("media_assets.image")
    elif "/trending" in path_lower:
        intents += ["trending.popular", "trending.top_rated"]
    elif "/review" in path_lower:
        intents.append("review.lookup")
    elif "/genre/movie" in path_lower:
        intents.append("genre.list.movie")
    elif "/genre/tv" in path_lower:
        intents.append("genre.list.tv")
    else:
        if "/movie" in path_lower:
            intents.append("details.movie")
        elif "/tv" in path_lower:
            intents.append("details.tv")
        elif "/person" in path_lower:
            intents.append("details.person")
        elif "/company" in path_lower:
            intents.append("company.details")
        elif "/network" in path_lower:
            intents.append("network.details")
        elif "/collection" in path_lower:
            intents.append("collection.details")

    if not intents:
        intents.append("miscellaneous")

    return intents

def _detect_entities(endpoint: str, parameters: list) -> list:
    entities = set()

    # Check query parameters
    for param in parameters:
        entity = PARAM_TO_ENTITY_MAP.get(param)
        if entity:
            entities.add(entity)

    # Check path placeholders
    for param, entity in PARAM_TO_ENTITY_MAP.items():
        if f"{{{param}}}" in endpoint:
            entities.add(entity)

    return list(entities)


def _detect_produced_entities(endpoint: str, parameters: list) -> list:
    produces = set()
    path = endpoint.lower()

    if "/discover/movie" in path:
        produces.add("movie")
    if "/discover/tv" in path:
        produces.add("tv")
    if "/person/" in path and "/movie_credits" in path:
        produces.add("movie")
    if "/person/" in path and "/tv_credits" in path:
        produces.add("tv")
    if "/search/person" in path:
        produces.add("person")
    if "/search/movie" in path:
        produces.add("movie")
    if "/search/tv" in path:
        produces.add("tv")
    if "/search/collection" in path:
        produces.add("collection")
    if "/search/company" in path:
        produces.add("company")
    if "/search/network" in path:
        produces.add("network")
    if "/search/keyword" in path:
        produces.add("keyword")
    if "/collection/" in path:
        produces.add("movie")

    return list(produces)


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
    parameters = obj.get("parameters", {})

    intents_detected = _detect_intents(endpoint_path, description)

    return {
        "path": endpoint_path,
        "description": description,
        "intents": json.dumps([{ "intent": i } for i in intents_detected]),  # <-- USE intents_detected
        "supported_intents": json.dumps(intents_detected),
        "media_type": "tv" if "/tv" in endpoint_path else "movie" if "/movie" in endpoint_path else "any",
        "consumes_entities": json.dumps(_detect_entities(endpoint_path, parameters.keys())),
        "produces_entities": json.dumps(_detect_produced_entities(endpoint_path, parameters.keys())),
        "supports_parameters": json.dumps(list(parameters))  # already fine
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

