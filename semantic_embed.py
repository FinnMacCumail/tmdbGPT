import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from tmdbv3api import TMDb, Genre, Company, Network, Person, Movie, TV
from dotenv import load_dotenv
from query_classifier import QueryClassifier
import re

load_dotenv()

class SemanticEmbedder:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path="./sec_intent_chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="tmdb_endpoints",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer(embedding_model)

        # Setup TMDb with API key validation
        self._init_tmdb()

        # Setup entity configuration
        self.entity_config = {
            "parameter_entities": self._get_parameter_entity_map(),
            "value_maps": self._initialize_value_maps(),
            "dynamic_resolvers": {
                "company": self._fetch_company_id,
                "network": self._fetch_network_id,
                "person": self._fetch_person_id
            }
        }

    def _init_tmdb(self):
        self.tmdb = TMDb()
        self.tmdb.api_key = os.getenv("NON_B_TMDB_API_KEY")
        assert self.tmdb.api_key, "❌ NON_B_TMDB_API_KEY is missing from environment!"

        self.genre = Genre(); self.genre.tmdb = self.tmdb
        self.company = Company(); self.company.tmdb = self.tmdb
        self.network = Network(); self.network.tmdb = self.tmdb
        self.person = Person(); self.person.tmdb = self.tmdb
        self.movie = Movie(); self.movie.tmdb = self.tmdb
        self.tv = TV(); self.tv.tmdb = self.tmdb

    def _get_parameter_entity_map(self) -> Dict:
        """Defines semantic types for all API parameters"""
        return {
            # ================== PATH PARAMETERS ==================
            "collection_id": "collection",
            "season_number": "season",
            "episode_number": "episode",
            "credit_id": "credit",
            "movie_id": "movie",
            "tv_id": "tv",
            "person_id": "person",
            "collection_id": "collection",
            "company_id": "company",
            "network_id": "network",
            "credit_id": "credit",
            "review_id": "review",
            "season_number": "season",
            "episode_number": "episode",
            
            # ================== CONTENT METADATA ==================
            "with_genres": "genre",
            "with_keywords": "keyword",
            "with_companies": "company",
            "with_networks": "network",
            "with_people": "person",
            "with_cast": "person",
            "with_crew": "person",
            "with_watch_providers": "provider",
            "with_status": "tv_status",
            "with_type": "tv_type",
            "with_watch_monetization_types": "monetization",        
            # ================== TEMPORAL FILTERS ==================
            "year": "year",
            "primary_release_year": "year",
            "first_air_date_year": "year",
            "air_date.gte": "date",
            "release_date.gte": "date",
            "primary_release_date.gte": "date",
            "first_air_date.gte": "date",
            
            # ================== GEOGRAPHIC/LOCALIZATION ==================
            "region": "country",
            "certification_country": "country",
            "watch_region": "country",
            "with_original_language": "language",
            
            # ================== CONTENT RATINGS ==================
            "certification": "rating",
            "certification.lte": "rating",
            "certification.gte": "rating",
            
            # ================== TECHNICAL SPECS ==================
            "with_runtime.gte": "duration",
            "with_runtime.lte": "duration",
            
            # ================== QUALITY METRICS ==================
            "vote_average.gte": "score",
            "vote_count.gte": "count",
            
            # ================== SEARCH PARAMETERS ==================
            "query": "search_term",
            "include_adult": "content_filter",
            "include_video": "media_type",
            
            # ================== PAGINATION/CONTROL ==================
            "page": "pagination",
            "timezone": "time_zone",
            
            # ================== SPECIALIZED FILTERS ==================
            "without_genres": "genre_exclusion",
            "without_keywords": "keyword_exclusion",
            "without_companies": "company_exclusion",
            "screened_theatrically": "release_format",
            "include_null_first_air_dates": "null_value_handling"
        }

    def _initialize_value_maps(self) -> Dict:
        return {
            "genre": self._fetch_genre_map(),
            "country": {'usa': 'US', 'france': 'FR'},
            "language": {'english': 'en', 'spanish': 'es'},
            "rating": {"G": "G", "PG": "PG"},
            "tv_status": {"returning": 0, "ended": 3},
            "monetization": {"stream": "flatrate"}
        }

    def _fetch_genre_map(self) -> Dict:
        try:
            return {g.name.lower(): g.id for g in self.genre.movie_list()}
        except Exception as e:
            print(f"⚠️ Failed to fetch genres: {str(e)}")
            return {'action': 28, 'drama': 18}

    def _fetch_company_id(self, company_name: str) -> int:
        try:
            companies = self.company.search(company_name)
            return companies[0].id if companies else None
        except Exception as e:
            print(f"⚠️ Failed to fetch company ID: {str(e)}")
            return None

    def _fetch_network_id(self, network_name: str) -> int:
        try:
            networks = self.network.search(network_name)
            return networks[0].id if networks else None
        except Exception as e:
            print(f"⚠️ Failed to fetch network ID: {str(e)}")
            return None

    def _fetch_person_id(self, person_name: str) -> int:
        try:
            results = self.person.search(person_name)
            if results:
                return results[0].id
            return None
        except Exception as e:
            print(f"⚠️ Failed to fetch person ID: {str(e)}")
            return {
                "tom cruise": 500,
                "meryl streep": 5064,
                "christopher nolan": 525
            }.get(person_name.lower())

    def _enrich_parameters(self, params: List) -> List:
        enriched = []
        for param in params:
            pname = param["name"]
            param_info = {
                "name": pname,
                "type": param["schema"].get("type", "string"),
                "required": param.get("required", False),
                "description": param.get("description", ""),
                "entity_type": self.entity_config["parameter_entities"].get(pname, "general"),
                "value_map": self._get_value_map(pname),
                "examples": self._generate_examples(pname)
            }
            if not isinstance(param_info["value_map"], dict):
                param_info["value_map"] = {}
            enriched.append(param_info)
        return enriched

    def _get_value_map(self, param_name: str) -> Dict:
        entity_type = self.entity_config["parameter_entities"].get(param_name)
        if entity_type in self.entity_config["value_maps"]:
            return self.entity_config["value_maps"][entity_type]
        if entity_type in self.entity_config["dynamic_resolvers"]:
            return self.entity_config["dynamic_resolvers"].get(entity_type, {})
        return {}

    def _generate_examples(self, param_name: str) -> List:
        examples = {
            "with_genres": ["action", "drama", "comedy"],
            "year": ["1999", "2005-2010", "2020"],
            "region": ["US", "FR", "JP"]
        }
        return examples.get(param_name, [])

    def _build_embedding_text(self, endpoint: str, metadata: Dict) -> str:
        components = [
            f"## API Endpoint: {endpoint}",
            f"**HTTP Method**: {metadata['method']}",
            f"**Summary**: {metadata['summary']}"
        ]
        if metadata["parameters"]:
            components.append("### Parameters:")
            for param in json.loads(metadata["parameters"]):
                desc = [
                    f"**{param['name']}** ({param['type']})",
                    f"- Required: {param['required']}",
                    f"- Entity Type: {param['entity_type']}"
                ]
                value_map = param.get("value_map", {})
                if isinstance(value_map, dict) and value_map:
                    examples = ", ".join([f"{k}→{v}" for k, v in list(value_map.items())[:3]])
                    desc.append(f"- Example Mappings: {examples}")
                components.append("\n".join(desc))
        if metadata["intents"]:
            components.append("### Supported Intents:")
            components.append("- " + "\n- ".join(json.loads(metadata["intents"])))
        return "\n\n".join(components)

    def process_endpoints(self, spec_path: str = "tmdb.json"):
        with open(spec_path, "r") as f:
            api_spec = json.load(f)

        embeddings, metadatas, ids = [], [], []

        for endpoint, details in api_spec["paths"].items():
            verb_info = details["get"]
            base_metadata = {
                "path": endpoint,
                "method": verb_info["method"],
                "summary": verb_info["summary"],
                "parameters": json.dumps(self._enrich_parameters(verb_info.get("parameters", [])), ensure_ascii=False),
                "intents": json.dumps(detect_intents(endpoint))
            }
            entity_types = list({
                p["entity_type"] for p in json.loads(base_metadata["parameters"])
                if p["entity_type"] != "general"
            })
            temp_metadata = {
                **base_metadata,
                "entity_types": json.dumps(entity_types)
            }
            embedding_text = self._build_embedding_text(endpoint, temp_metadata)

            full_metadata = {
                **temp_metadata,
                "embedding_text": embedding_text
            }
            embedding_text = self._build_embedding_text(endpoint, full_metadata)
            embedding = self.embedder.encode(embedding_text).tolist()
            ids.append(endpoint)
            embeddings.append(embedding)
            metadatas.append(full_metadata)

        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

def detect_intents(endpoint: str) -> List[str]:
    matched = []
    for intent, data in QueryClassifier.INTENT_MAP.items():
        for pattern in data["endpoints"]:
            regex = re.sub(r"{[^}]+}", r"([^/]+)", pattern) + "$"
            if re.fullmatch(regex, endpoint):
                matched.append(intent)
                break
    return list(set(matched))

if __name__ == "__main__":
    embedder = SemanticEmbedder()
    embedder.process_endpoints(spec_path="data/tmdb.json")