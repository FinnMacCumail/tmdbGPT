import json
from sentence_transformers import SentenceTransformer
import chromadb

class SemanticEmbedder:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("tmdb_endpoints")

    def _detect_intents(self, endpoint, parameters):
        if "/recommendations" in endpoint or "/similar" in endpoint:
            return [{"intent": "recommendation", "confidence": 1.0}]
        if "/movie/" in endpoint and endpoint.count("{") == 1:
            return [{"intent": "details.movie", "confidence": 1.0}]
        if "/tv/" in endpoint and endpoint.count("{") == 1:
            return [{"intent": "details.tv", "confidence": 1.0}]
        if endpoint.startswith("/discover/"):
            return [{"intent": "discovery.filtered", "confidence": 1.0}]
        if endpoint.startswith("/trending/"):
            return [{"intent": "trending.popular", "confidence": 1.0}]
        if endpoint.startswith("/search/movie"):
            return [{"intent": "search.movie", "confidence": 1.0}]
        if endpoint.startswith("/search/person"):
            return [{"intent": "search.person", "confidence": 1.0}]
        if endpoint.startswith("/search/tv"):
            return [{"intent": "search.tv", "confidence": 1.0}]
        if endpoint.startswith("/search/company"):
            return [{"intent": "search.company", "confidence": 1.0}]
        if endpoint.startswith("/search/collection"):
            return [{"intent": "search.collection", "confidence": 1.0}]
        if "credits" in endpoint:
            return [{"intent": "credits.person", "confidence": 1.0}]
        if "/collection/" in endpoint:
            return [{"intent": "collection.details", "confidence": 1.0}]
        if "/company/" in endpoint:
            return [{"intent": "company.details", "confidence": 1.0}]
        if "/network/" in endpoint:
            return [{"intent": "network.details", "confidence": 1.0}]
        if "/movie/" in endpoint and "keywords" in endpoint:
            return [{"intent": "keywords.movie", "confidence": 1.0}]
        if "/credit/" in endpoint:
            return [{"intent": "credits.lookup", "confidence": 1.0}]
        if endpoint.startswith("/genre/movie/list"):
            return [{"intent": "genre.list.movie", "confidence": 1.0}]
        if endpoint.startswith("/genre/tv/list"):
            return [{"intent": "genre.list.tv", "confidence": 1.0}]
        if endpoint.startswith("/review/"):
            return [{"intent": "review.lookup", "confidence": 1.0}]
        return [{"intent": "miscellaneous", "confidence": 0.3}]

    def _detect_entities(self, parameters, endpoint):
        param_names = [p["name"] for p in parameters if isinstance(p, dict)]
        detected = []
        if "with_people" in param_names or "person_id" in endpoint:
            detected.append("person")
        if "with_genres" in param_names or "genre_id" in endpoint:
            detected.append("genre")
        if "with_companies" in param_names:
            detected.append("company")
        if "movie_id" in endpoint:
            detected.append("movie")
        if "tv_id" in endpoint:
            detected.append("tv")
        if "collection_id" in endpoint:
            detected.append("collection")
        if "company_id" in endpoint:
            detected.append("company")
        if "network_id" in endpoint:
            detected.append("network")
        if "credit_id" in endpoint:
            detected.append("credit")
        if "review_id" in endpoint:
            detected.append("review")
        return detected

    def _detect_produced_entities(self, endpoint):
        produced = []
        if "credits" in endpoint:
            if "/movie/" in endpoint:
                produced.append("movie_id")
            if "/tv/" in endpoint:
                produced.append("tv_id")
            if "/person/" in endpoint:
                produced.extend(["movie_id", "tv_id"])
        if "/recommendations" in endpoint or "/similar" in endpoint:
            if "/movie/" in endpoint:
                produced.append("movie_id")
            if "/tv/" in endpoint:
                produced.append("tv_id")
        if "/collection/" in endpoint:
            produced.append("collection_id")
        if "/company/" in endpoint:
            produced.append("company_id")
        if "/network/" in endpoint:
            produced.append("network_id")
        if "/genre/" in endpoint:
            produced.append("genre_id")
        if "/keyword/" in endpoint:
            produced.append("keyword_id")
        if "/review/" in endpoint:
            produced.append("review_id")
        if "/credit/" in endpoint:
            produced.append("credit_id")
        return list(set(produced))

    def _create_embedding_text(self, endpoint, details):
        desc = details.get("description", "")

        # ✅ Enrich description for discover/movie
        if endpoint == "/discover/movie":
            desc += (
                "\nThis endpoint lets you find movies with flexible filters."
                "\nSupports `with_people` to filter by multiple actors (e.g., Robert De Niro and Al Pacino),"
                "\n`with_genres` for genre-based filtering, and"
                "\n`primary_release_year` for temporal queries."
                "\nPerfect for discovery use cases and multi-person cast matching."
            )

        return f"Endpoint: {endpoint}\nDescription: {desc.strip()}"

    def _create_metadata(self, endpoint, details):
        param_names = [p["name"] for p in details.get("parameters", []) if isinstance(p, dict)]
        return {
            "path": endpoint,
            "param_names": json.dumps(param_names),
            "intents": json.dumps(self._detect_intents(endpoint, details.get("parameters", []))),
            "entities": ", ".join(self._detect_entities(details.get("parameters", []), endpoint)),
            "consumes_entities": json.dumps([p["name"] for p in details.get("parameters", []) if p.get("in") == "path"]),
            "produces_entities": json.dumps(self._detect_produced_entities(endpoint))
        }

    def process_endpoints(self):
        with open("data/tmdb.json") as f:
            schema = json.load(f)

        ids, docs, metas, embs = [], [], [], []

        for path, methods in schema.get("paths", {}).items():
            get_method = methods.get("get")
            if not get_method:
                continue

            embedding_text = self._create_embedding_text(path, get_method)
            metadata = self._create_metadata(path, get_method)
            embedding = self.embedder.encode(embedding_text).tolist()

            ids.append(path)
            docs.append(embedding_text)
            metas.append(metadata)
            embs.append(embedding)

        print(f"📦 Upserting {len(ids)} endpoints into ChromaDB (tmdb_endpoints)...")
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        print("✅ Done.")

if __name__ == "__main__":
    SemanticEmbedder().process_endpoints()
