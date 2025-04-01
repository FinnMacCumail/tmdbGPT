import json
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from tmdbv3api import TMDb, Genre, Company, Network, Person, Movie, TV
from dotenv import load_dotenv

load_dotenv()

class SemanticEmbedder:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path="./sec_intent_chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="tmdb_endpoints",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer(embedding_model)
        self._init_tmdb()
        self.entity_config = self._initialize_entity_config()

    def _init_tmdb(self):
        self.tmdb = TMDb()
        self.tmdb.api_key = os.getenv("NON_B_TMDB_API_KEY")
        assert self.tmdb.api_key, "❌ NON_B_TMDB_API_KEY missing from environment!"
        
        self.genre = Genre()
        self.company = Company()
        self.network = Network()
        self.person = Person()
        self.movie = Movie()
        self.tv = TV()

    def _initialize_entity_config(self) -> Dict:
        return {
            "parameter_entities": self._get_parameter_entity_map(),
            "value_maps": self._initialize_value_maps(),
            "dynamic_resolvers": {
            # Core Entities
            "movie": self._fetch_movie_id,
            "tv": self._fetch_tv_id,
            "person": self._fetch_person_id,
            
            # Organizational Entities
            "company": self._fetch_company_id,
            "network": self._fetch_network_id,
            "collection": self._fetch_collection_id,
            
            # Content Entities
            "keyword": self._fetch_keyword_id
        }
        }

    def _fetch_genre_map(self) -> Dict:
        """Fetch genre mappings from TMDB API with error handling"""
        try:
            movie_genres = {g.name.lower(): g.id for g in self.genre.movie_list()}
            tv_genres = {g.name.lower(): g.id for g in self.genre.tv_list()}
            return {**movie_genres, **tv_genres}
        except Exception as e:
            print(f"⚠️ Failed to fetch genres: {str(e)}")
            return {'action': 28, 'drama': 18}  # Fallback values
    
    def _fetch_company_id(self, company_name: str) -> int:
        """Resolve company name to ID"""
        try:
            companies = self.company.search(company_name)
            return companies[0].id if companies else None
        except Exception as e:
            print(f"⚠️ Failed to fetch company ID: {str(e)}")
            return None

    def _fetch_network_id(self, network_name: str) -> int:
        """Resolve network name to ID"""
        try:
            networks = self.network.search(network_name)
            return networks[0].id if networks else None
        except Exception as e:
            print(f"⚠️ Failed to fetch network ID: {str(e)}")
            return None

    def _fetch_person_id(self, person_name: str) -> int:
        """Resolve person name to ID"""
        try:
            results = self.person.search(person_name)
            return results[0].id if results else None
        except Exception as e:
            print(f"⚠️ Failed to fetch person ID: {str(e)}")
            return None       
        
    def _fetch_movie_id(self, title: str) -> int:
        results = self.movie.search(title)
        return results[0].id if results else None

    def _fetch_tv_id(self, title: str) -> int:
        results = self.tv.search(title)
        return results[0].id if results else None

    def _fetch_collection_id(self, name: str) -> int:
        results = self.movie.search_collection(name)
        return results[0].id if results else None

    def _fetch_keyword_id(self, keyword: str) -> int:
        results = self.movie.search_keyword(keyword)
        return results[0].id if results else None

    def resolve_review_id(self, movie_id: int, review_text: str) -> int:
        """Get review ID from movie context"""
        reviews = self.movie.reviews(movie_id)
        return next((r.id for r in reviews if review_text in r.content), None)

    def resolve_credit_id(self, person_id: int, media_id: int) -> int:
        """Get credit ID from person/media context"""
        credits = self.person.combined_credits(person_id)
        return next((c.id for c in credits if c.media_id == media_id), None)

    def _get_parameter_entity_map(self) -> Dict:
        """Core parameter-to-entity mappings for resolution"""
        return {
        # ======== Path Parameters ========
        "movie_id": "movie",
        "tv_id": "tv",
        "person_id": "person",
        "company_id": "company",
        "network_id": "network",
        "collection_id": "collection",
        "keyword_id": "keyword",
        "review_id": "review",
        "credit_id": "credit",
        
        # ======== Query Parameters ========
        "with_cast": "person",
        "with_crew": "person",
        "with_companies": "company",
        "with_networks": "network",
        "with_collections": "collection",
        "with_keywords": "keyword",
        "with_genres": "genre",
        "region": "country",
        "with_original_language": "language"
    }

    def _initialize_value_maps(self) -> Dict:
        return {
            "genre": self._fetch_genre_map(),
            "country": {'usa': 'US', 'france': 'FR'},
            "language": {'english': 'en', 'spanish': 'es'},
            "rating": {"G": "G", "PG": "PG"}
        }

    def _build_embedding_text(self, endpoint: str, metadata: Dict) -> str:
        """Focus on semantic-relevant fields only"""
        components = [
            f"API Endpoint: {metadata['path']}",
            f"Description: {metadata['description']}",
            f"Parameters: {self._format_parameters(metadata['parameters'])}"
        ]
        return "\n\n".join(components)

    def _format_parameters(self, params_str: str) -> str:
        """Handle parameter data as serialized JSON"""
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError:
            return "No parameters"
        
        param_strings = []
        for param in params:
            desc = [
                f"Parameter: {param.get('name', 'unnamed')}",
                f"Type: {param.get('type', 'unknown')}",
                f"Description: {param.get('description', 'No description')}"
            ]
            param_strings.append("\n".join(desc))
        return "\n---\n".join(param_strings)

    def process_endpoints(self, spec_path: str = "tmdb.json"):
        with open(spec_path, "r") as f:
            api_spec = json.load(f)

        embeddings, metadatas, ids = [], [], []

        for endpoint, details in api_spec["paths"].items():
            verb_info = details["get"]
            enriched_params = self._enrich_parameters(verb_info.get("parameters", []))
            
            metadata = {
                "path": endpoint,
                "description": verb_info.get("description", ""),
                "parameters": json.dumps(enriched_params),  # Keep as JSON string
                "entity_types": json.dumps(list({
                    p["entity_type"] for p in enriched_params  # Use original list here
                    if p["entity_type"] != "general"
                }))
            }

            embedding_text = self._build_embedding_text(endpoint, metadata)
            embedding = self.embedder.encode(embedding_text).tolist()
            
            ids.append(endpoint)
            embeddings.append(embedding)
            metadatas.append(metadata)

        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def _enrich_parameters(self, params: List) -> List:
        """Essential parameter enrichment for semantic search"""
        return [{
            "name": param["name"],
            "type": param.get("schema", {}).get("type", "string"),
            "description": param.get("description", ""),
            "entity_type": self.entity_config["parameter_entities"].get(
                param["name"], "general"
            )
        } for param in params]    

if __name__ == "__main__":
    embedder = SemanticEmbedder()
    embedder.process_endpoints(spec_path="data/tmdb.json")