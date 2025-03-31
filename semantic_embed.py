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

# Initialize TMDB API
tmdb = TMDb()
tmdb.api_key = os.getenv("TMDB_API_KEY")
tmdb.language = 'en'

class SemanticEmbedder:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path="./sec_intent_chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="tmdb_endpoints",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer(embedding_model)
        # Initialize TMDB API components
        # self.tmdb = TMDb()
        # self.tmdb.api_key = os.getenv("TMDB_API_KEY")
        #self.tmdb.language = 'en'
        self.genre = Genre()
        self.company = Company()
        self.network = Network()
        self.person = Person()
        self.movie = Movie()
        self.tv = TV()
        self.entity_config = {
            "parameter_entities": self._get_parameter_entity_map(),
            "value_maps": self._initialize_value_maps(),
            "dynamic_resolvers": {
                "company": self._fetch_company_map(),
                "network": self._fetch_network_map(),
                "person": self._fetch_person_map()
            }
        }
        
        # Entity resolution mappings
        self.entity_config = {
            "parameter_entities": self._get_parameter_entity_map(),
            "value_maps": self._initialize_value_maps(),
            "dynamic_resolvers": {
                "company": self._fetch_company_id,
                "network": self._fetch_network_id,
                "person": self._fetch_person_id
            }
        }

    def _get_parameter_entity_map(self) -> Dict:
        """Defines semantic types for all API parameters"""
        return {
            # Content metadata
            "with_genres": "genre",
            "with_keywords": "keyword",
            "with_companies": "company",
            "with_networks": "network",
            
            # Temporal filters
            "year": "year",
            "primary_release_year": "year",
            "first_air_date_year": "year",
            "air_date.gte": "date",
            "release_date.gte": "date",
            
            # Geographic/Localization
            "region": "country",
            "certification_country": "country",
            "with_original_language": "language",
            
            # Content ratings
            "certification": "rating",
            "certification.lte": "rating",
            
            # Technical specs
            "with_runtime.gte": "duration",
            "with_watch_providers": "provider",
            
            # Quality metrics
            "vote_average.gte": "score",
            "vote_count.gte": "count",
            
            # TV specific
            "with_status": "tv_status",
            "with_type": "tv_type",
            
            # Special filters
            "with_watch_monetization_types": "monetization"
        }

    def _initialize_value_maps(self) -> Dict:
        """Prepopulated and dynamic value mappings"""
        return {
            "genre": self._fetch_genre_map(),
            "country": {'usa': 'US', 'france': 'FR'},
            "language": {'english': 'en', 'spanish': 'es'},
            "rating": {"G": "G", "PG": "PG"},
            "tv_status": {"returning": 0, "ended": 3},
            "monetization": {"stream": "flatrate"}
        }

    def _fetch_genre_map(self) -> Dict:  # Correct method name
        """Fetch genre ID mappings"""
        try:
            return {g.name.lower(): g.id for g in self.genre.movie_list()}
        except Exception as e:
            print(f"⚠️ Failed to fetch genres: {str(e)}")
            return {'action': 28, 'drama': 18}

    # CORRECTED RESOLVER METHODS (now return full maps)
    def _fetch_company_map(self) -> Dict:
        """Fetch companies using movie details endpoint"""
        try:
            # Get details for a known movie with companies
            movie = self.movie.details(299536)  # Avengers: Infinity War
            return {c.name.lower(): c.id for c in movie.production_companies}
        except Exception as e:
            print(f"⚠️ Failed to fetch companies: {str(e)}")
            return {'marvel studios': 420, 'pixar': 3}
        
    def _fetch_network_map(self) -> Dict:
        """Fetch networks using TV details endpoint"""
        try:
            # Get details for a known TV show
            show = self.tv.details(1399)  # Game of Thrones
            return {n.name.lower(): n.id for n in show.networks}
        except Exception as e:
            print(f"⚠️ Failed to fetch networks: {str(e)}")
            return {'hbo': 49, 'netflix': 213}
        except Exception as e:
            print(f"⚠️ Failed to fetch networks: {str(e)}")
            return {'hbo': 49, 'netflix': 213}

    def _fetch_person_map(self) -> Dict:
        """Fetch popular people with proper data handling"""
        try:
            people = self.person.popular()
            return {p.name.lower(): p.id for p in people}
        except Exception as e:
            print(f"⚠️ Failed to fetch people: {str(e)}")
            return {'tom cruise': 500, 'meryl streep': 5064}
    
    def _fetch_country_map(self) -> Dict:
        """ISO 3166-1 country codes"""
        return {'usa': 'US', 'france': 'FR', 'germany': 'DE'}

    def _fetch_language_map(self) -> Dict:
        """ISO 639-1 language codes"""
        return {'english': 'en', 'french': 'fr', 'spanish': 'es'}

    def _fetch_company_id(self, company_name: str) -> int:
        """Fetch company ID by name"""
        try:
            companies = self.company.search(company_name)
            return companies[0].id if companies else None
        except Exception as e:
            print(f"⚠️ Failed to fetch company ID: {str(e)}")
            return None
        
    def _fetch_person_id(self, person_name: str) -> int:
        """Fetch TMDB person ID by name with fallback"""
        try:
            results = self.person.search(person_name)
            if results:
                return results[0].id
            return None
        except Exception as e:
            print(f"⚠️ Failed to fetch person ID: {str(e)}")
            # Fallback to manual mapping
            manual_map = {
                "tom cruise": 500,
                "meryl streep": 5064,
                "christopher nolan": 525
            }
            return manual_map.get(person_name.lower())

    def _fetch_network_id(self, network_name: str) -> int:
        """Fetch network ID by name"""
        try:
            networks = self.network.search(network_name)
            return networks[0].id if networks else None
        except Exception as e:
            print(f"⚠️ Failed to fetch network ID: {str(e)}")
            return None
    
    
    def _fetch_dynamic_data(self, endpoint: str) -> Dict:
        """Fetch API-driven value mappings"""
        try:
            if "genre" in endpoint:
                return {g.name.lower(): g.id for g in Genre().movie_list()}
            if "company" in endpoint:
                return {c.name.lower(): c.id for c in Company().popular()}
            if "network" in endpoint:
                return {n.name.lower(): n.id for n in Network().popular()}
        except Exception as e:
            print(f"⚠️ Failed to fetch {endpoint}: {str(e)}")
            return {}

    def _enrich_parameters(self, params: List) -> List:
        """Ensure value maps are always dictionaries"""
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
            
            # Ensure value_map is always a dict
            if not isinstance(param_info["value_map"], dict):
                param_info["value_map"] = {}
                
            enriched.append(param_info)
        return enriched

    def _get_value_map(self, param_name: str) -> Dict:
        """Get appropriate value map for parameter"""
        entity_type = self.entity_config["parameter_entities"].get(param_name)
        
        if entity_type in self.entity_config["value_maps"]:
            return self.entity_config["value_maps"][entity_type]
            
        if entity_type in self.entity_config["dynamic_resolvers"]:
            # Return pre-fetched map directly
            return self.entity_config["dynamic_resolvers"].get(entity_type, {})
            
        return {}

    def _generate_examples(self, param_name: str) -> List:
        """Create usage examples for parameters"""
        examples = {
            "with_genres": ["action", "drama", "comedy"],
            "year": ["1999", "2005-2010", "2020"],
            "region": ["US", "FR", "JP"]
        }
        return examples.get(param_name, [])

    def _build_embedding_text(self, endpoint: str, metadata: Dict) -> str:
        """Safe handling of value maps"""
        components = [
            f"## API Endpoint: {endpoint}",
            f"**HTTP Method**: {metadata['method']}",
            f"**Summary**: {metadata['summary']}"
        ]

        if metadata["parameters"]:
            components.append("### Parameters:")
            for param in json.loads(metadata["parameters"]):
                param_desc = [
                    f"**{param['name']}** ({param['type']})",
                    f"- Required: {param['required']}",
                    f"- Entity Type: {param['entity_type']}"
                ]
                
                # Safe value map handling
                value_map = param.get("value_map", {})
                if isinstance(value_map, dict) and value_map:
                    examples = ", ".join([f"{k}→{v}" for k,v in list(value_map.items())[:3]])
                    param_desc.append(f"- Example Mappings: {examples}")
                
                components.append("\n".join(param_desc))

        if metadata["intents"]:
            components.append("### Supported Intents:")
            components.append("- " + "\n- ".join(json.loads(metadata["intents"])))

        return "\n\n".join(components)
    
    def process_endpoints(self, spec_path: str = "tmdb.json"):
        """Main processing pipeline"""
        with open(spec_path, "r") as f:
            api_spec = json.load(f)

        embeddings = []
        metadatas = []
        ids = []

        for endpoint, details in api_spec["paths"].items():
            verb_info = details["get"]
            
            # First build the base metadata
            base_metadata = {
                "path": endpoint,
                "method": verb_info["method"],
                "summary": verb_info["summary"],
                "parameters": json.dumps(
                    self._enrich_parameters(verb_info.get("parameters", [])),
                    ensure_ascii=False
                ),
                "intents": json.dumps(detect_intents(endpoint))
            }

            # Now calculate entity types using the already-created parameters
            entity_types = list({
                p["entity_type"] for p in json.loads(base_metadata["parameters"])
                if p["entity_type"] != "general"
            })
            
            # Add entity_types to the metadata
            full_metadata = {**base_metadata, "entity_types": json.dumps(entity_types)}

            # Generate embedding text
            embedding_text = self._build_embedding_text(endpoint, full_metadata)
            embedding = self.embedder.encode(embedding_text).tolist()

            ids.append(endpoint)
            embeddings.append(embedding)
            metadatas.append(full_metadata)  # Use the full metadata

        # Batch upsert to Chroma
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

# Helper functions (implement according to your API)
def detect_intents(endpoint: str) -> List[str]:
    """Match endpoints to intents using pattern matching"""
    matched_intents = []
    
    for intent_name, intent_data in QueryClassifier.INTENT_MAP.items():
        for pattern in intent_data["endpoints"]:
            # Convert OpenAPI path to regex
            regex_pattern = re.sub(r"{[^}]+}", r"([^/]+)", pattern) + "$"
            if re.fullmatch(regex_pattern, endpoint):
                matched_intents.append(intent_name)
                break  # No need to check other patterns for this intent
    
    return list(set(matched_intents))  # Deduplicate

if __name__ == "__main__":
    embedder = SemanticEmbedder()
    embedder.process_endpoints(spec_path="data/tmdb.json")