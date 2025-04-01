import json
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from tmdbv3api import TMDb, Genre, Company, Network, Person, Movie, TV
from dotenv import load_dotenv
from query_classifier import QueryClassifier 
import inflect

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
        self.query_classifier = QueryClassifier()
        # Initialize description patterns FIRST
        self.description_patterns = self._init_description_patterns() 
        # Then create entity config that depends on it
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

    def _init_description_patterns(self) -> Dict:
        """Enhanced description patterns with priority levels"""
        return {
            "country": {"patterns": [r"iso\s*3166", r"country\s*code"], "priority": 1},
            "person": {"patterns": [r"actor", r"crew", r"director"], "priority": 2},
            "date": {"patterns": [r"date", r"year", r"release"], "priority": 3},
            "genre": {"patterns": [r"genre", r"category"], "priority": 2},
            "language": {"patterns": [r"language", r"locale"], "priority": 3}
        }

    def _initialize_entity_config(self) -> Dict:
        return {
            "parameter_entities": self._get_enhanced_parameter_map(),
            "value_maps": self._initialize_value_maps(),
            "dynamic_resolvers": {
                "movie": self._fetch_movie_id,
                "tv": self._fetch_tv_id,
                "person": self._fetch_person_id,
                "company": self._fetch_company_id,
                "network": self._fetch_network_id,
                "collection": self._fetch_collection_id,
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
  
    def _get_enhanced_parameter_map(self) -> Dict:
        """Hybrid mapping with path parameter analysis"""
        p = inflect.engine()
        entity_map = {}
        
        with open("data/tmdb.json") as f:
            spec = json.load(f)
        
        # First pass: Process path parameters
        for endpoint in spec["paths"]:
            # Extract hierarchical entities from path
            path_segments = [seg for seg in endpoint.split('/') if seg]
            for seg in path_segments:
                if '{' in seg:
                    param = seg[1:-1]  # Remove curly braces
                    clean_entity = param.replace('_id', '')
                    entity_map[param] = clean_entity
                    # Add parent-child relationships
                    parent = '/'.join(path_segments[:path_segments.index(seg)])
                    if parent:
                        entity_map[f"{parent}.{clean_entity}"] = clean_entity

        # Second pass: Process query parameters
        for endpoint, details in spec["paths"].items():
            params = details["get"].get("parameters", [])
            for param in params:
                pname = param["name"]
                entity = None
                
                # 1. Check explicit path-derived mappings
                if pname in entity_map:
                    entity = entity_map[pname]
                
                # 2. Pattern-based matching
                if not entity:
                    for pattern, resolver in [
                        #(r"with_(.+)", lambda x: x.group(1).rstrip('s')),
                        (r"with_(.+)", lambda x: p.singular_noun(x.group(1)) or x.group(1)),
                        (r"(.+)_id$", lambda x: x.group(1)),
                        (r"(.+)_year$", "date")
                    ]:
                        match = re.fullmatch(pattern, pname)
                        if match:
                            entity = resolver(match) if callable(resolver) else resolver
                            break
                
                # 3. Description analysis fallback
                if not entity:
                    desc = param.get("description", "").lower()
                    for ent_type, config in self.description_patterns.items():
                        if any(re.search(p, desc) for p in config["patterns"]):
                            entity = ent_type
                            break
                
                entity_map[pname] = entity or "general"

        return entity_map


    def _initialize_value_maps(self) -> Dict:  # ENHANCED
        """Expanded value mappings with error handling"""
        return {
            "genre": self._fetch_genre_map(),
            "country": {'usa': 'US', 'france': 'FR', 'germany': 'DE'},
            "language": {'english': 'en', 'spanish': 'es', 'french': 'fr'},
            "rating": {"G": "G", "PG": "PG", "PG-13": "PG-13", "R": "R"},
            "provider": {"netflix": 8, "prime": 9}  # NEW
        }    
    
    def _build_embedding_text(self, endpoint: str, metadata: Dict) -> str:
        """Enhanced text generation with hierarchy markers"""
        components = [
            f"API Endpoint: {metadata['path']}",
            f"Description: {metadata['description']}",
            f"Entity Hierarchy: {' > '.join(json.loads(metadata['entity_types']))}",
            f"Core Intents: {', '.join(json.loads(metadata['intents']))}"
        ]
        
        params = json.loads(metadata["parameters"])
        if params:
            components.append("Parameters:")
            for param in params:
                components.append(
                    f"{param['name']} ({param['type']}): {param['entity_type']}"
                )
        
        return "\n\n".join(components)

    def process_endpoints(self, spec_path: str = "data/tmdb.json"):
        with open(spec_path, "r") as f:
            api_spec = json.load(f)

        embeddings, metadatas, ids = [], [], []

        for endpoint, details in api_spec["paths"].items():
            verb_info = details["get"]
            enriched_params = self._enrich_parameters(verb_info.get("parameters", []))
            
            # Generate hierarchical entities from path
            path_entities = []
            for seg in endpoint.split('/'):
                if '{' in seg:
                    clean_seg = seg[1:-1].replace('_id', '')
                    path_entities.append(clean_seg)
            
            # Combine with parameter-derived entities
            param_entities = list({p["entity_type"] for p in enriched_params if p["entity_type"] != "general"})
            all_entities = list(set(path_entities + param_entities))
            
            metadata = {
                "path": endpoint,
                "description": verb_info.get("description", ""),
                "parameters": json.dumps(enriched_params),
                "entity_types": json.dumps(all_entities),  # Now never empty
                "intents": json.dumps(self._detect_intents(endpoint, enriched_params))
            }

            embedding_text = self._build_embedding_text(endpoint, metadata)
            embedding = self.embedder.encode(embedding_text).tolist()
            
            ids.append(endpoint)
            embeddings.append(embedding)
            metadatas.append(metadata)

        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def _detect_intents(self, endpoint: str, params: List) -> List:
        """Enhanced intent detection with path analysis"""
        intents = []
        
        # Path-based intents
        if '/search/' in endpoint:
            intents.append("search")
        if '/discover/' in endpoint:
            intents.append("discovery")
        if '/similar' in endpoint:
            intents.append("recommendation")
            
        # Parameter-based intents
        param_entities = [p["entity_type"] for p in params]
        if "country" in param_entities:
            intents.append("regional")
        if "date" in param_entities:
            intents.append("temporal")
            
        return list(set(intents))


    def _enrich_parameters(self, params: List) -> List:
        """Enhanced parameter enrichment with entity resolution"""
        enriched = []
        for param in params:
            pname = param["name"]
            schema = param.get("schema", {})
            
            param_info = {
                "name": pname,
                "type": schema.get("type", "string"),
                "required": param.get("required", False),
                "description": param.get("description", ""),
                "entity_type": self._resolve_entity_type(pname),
                "constraints": {
                    "min": schema.get("minimum"),
                    "max": schema.get("maximum"),
                    "pattern": schema.get("pattern"),
                    "enum": schema.get("enum", [])
                },
                "resolved_values": self._get_resolved_values(pname)
            }

            # Add numeric subtype
            if param_info["type"] in ["integer", "number"]:
                param_info["entity_type"] += ".numeric"

            enriched.append(param_info)
        return enriched   

    def _resolve_entity_type(self, param_name: str) -> str:
        """Resolve entity with hierarchy awareness"""
        base_entity = self.entity_config["parameter_entities"].get(param_name, "general")
        
        # Split compound entities (e.g., tv.season)
        if '.' in base_entity:
            parts = base_entity.split('.')
            return '.'.join([p.replace('_id', '') for p in parts])
        
        return base_entity.replace('_id', '')

    def _get_resolved_values(self, param_name: str) -> Dict:
        """Get pre-resolved values for parameter"""
        entity_type = self.entity_config["parameter_entities"].get(param_name)
        return self.entity_config["value_maps"].get(entity_type, {})

if __name__ == "__main__":
    embedder = SemanticEmbedder()
    embedder.process_endpoints(spec_path="data/tmdb.json")