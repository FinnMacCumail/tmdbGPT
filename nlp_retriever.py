import json
import json5
import chromadb
import requests
import os
import spacy
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
import traceback
from requests.exceptions import RequestException
from utils.metadata_parser import MetadataParser
import networkx as nx
import time
from collections import defaultdict
from tabulate import tabulate
import uuid
from plan_validator import PlanValidator
from prompt_templates import PLAN_PROMPT 
from fallback_handler import FallbackHandler
from prompt_templates import PROMPT_TEMPLATES, DEFAULT_TEMPLATE
#from query_classifier import QueryClassifier

from param_resolver import ParamResolver
from llm_client import OpenAILLMClient
from dependency_manager import DependencyManager
from query_classifier import QueryClassifier


# Load API keys
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

# Load NLP and embedding models
nlp = spacy.load("en_core_web_trf")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# TMDB API configuration
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TMDB_API_KEY}"}


class EnhancedIntentAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.genre_map = self._fetch_genre_mappings()
        self.media_types = ["movie", "tv", "film", "show", "series"]
        self._add_custom_patterns()

    def _fetch_genre_mappings(self) -> Dict[str, int]:
        """Fetch genre IDs from TMDB with fallback"""
        try:
            response = requests.get(f"{BASE_URL}/genre/movie/list", headers=HEADERS)
            response.raise_for_status()
            return {
                genre["name"].lower(): genre["id"]
                for genre in response.json().get("genres", [])
            }
        except Exception as e:
            print(f"⚠️ Failed to fetch genres, using fallback: {str(e)}")
            return {
                "action": 28, "comedy": 35, "drama": 18,
                "horror": 27, "sci-fi": 878, "romance": 10749,
                "thriller": 53, "documentary": 99
            }

    def _add_custom_patterns(self):
        """Add custom entity recognition patterns"""
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            # Temporal patterns
            {"label": "DATE", "pattern": [{"LOWER": {"IN": ["now", "current", "recent"]}}]},
            {"label": "DATE", "pattern": [{"LOWER": "right"}, {"LOWER": "now"}]},

            # Media type patterns
            {"label": "MEDIA_TYPE", "pattern": [{"LEMMA": {"IN": self.media_types}}]},

            # Popularity metrics
            {"label": "POPULARITY", "pattern": [{"LEMMA": {"IN": ["popular", "trending"]}}]},

            # Genre patterns
            {"label": "GENRE", "pattern": [{"LOWER": {"IN": list(self.genre_map.keys())}}]},
        ]
        ruler.add_patterns(patterns)

    def extract_entities(self, query: str) -> Dict[str, List]:
        """Main entity extraction method"""
        doc = self.nlp(query)
        entities = defaultdict(list)
        
        # Process recognized entities
        for ent in doc.ents:
            self._process_core_entity(ent, entities)
        
        # Additional processing
        self._detect_comparative_phrases(doc, entities)
        self._extract_numeric_constraints(doc, entities)
        self._detect_media_type(doc, entities)
        self._extract_temporal_references(doc, entities)
        
        # Add intent-aware entity resolution
        self._enrich_with_intent_context(entities, doc)
        return dict(entities)
    
    def _enrich_with_intent_context(self, entities: Dict, doc):
        """Enhance entities based on query structure"""
        # Detect "most X" patterns
        for token in doc:
            if token.dep_ == 'amod' and token.head.dep_ == 'nsubj':
                if token.text in ['popular', 'trending']:
                    entities.setdefault('ranking_metric', []).append(token.text)
                    
        # Detect comparison patterns
        if any(t.lemma_ in ['compare', 'versus'] for t in doc):
            entities['comparison'] = [True]

    def _process_core_entity(self, ent, entities):
        """Process spaCy entities"""
        label = ent.label_
        if label == "PERSON":
            entities["person"].append(ent.text)
        elif label == "WORK_OF_ART":
            entities["title"].append(ent.text)
        elif label == "GPE":
            entities["region"].append(ent.text)
        elif label == "DATE":
            entities["date"].append(ent.text)
        elif label == "GENRE":
            entities["genre"].append(ent.text)
        elif label == "MEDIA_TYPE":
            entities["media_type"].append(ent.text)
        elif label == "POPULARITY":
            entities["popularity"].append(ent.text)

    def _detect_comparative_phrases(self, doc, entities):
        """Detect phrases like 'most popular'"""
        for token in doc:
            if token.text.lower() in ["most", "best", "top"] and token.head.pos_ == "ADJ":
                entities["comparative"].append(f"{token.text} {token.head.text}")

    def _extract_numeric_constraints(self, doc, entities):
        """Extract numeric ranges and comparisons"""
        for token in doc:
            if token.like_num or token.text in (">", "<", ">=", "<="):
                next_token = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if next_token and next_token.like_num:
                    entities["numeric_constraints"].append(f"{token.text}{next_token.text}")

    def _detect_media_type(self, doc, entities):
        """Detect implicit media type context"""
        if not entities.get("media_type"):
            media_terms = [t.text for t in doc if t.lemma_ in self.media_types]
            if media_terms:
                entities["media_type"].extend(media_terms)
            elif "movie" in doc.text.lower():
                entities["media_type"].append("movie")

    def _extract_temporal_references(self, doc, entities):
        """Extract time-related phrases"""
        for chunk in doc.noun_chunks:
            text = chunk.text.lower()
            if any(kw in text for kw in ["this week", "this month", "current"]):
                entities["temporal"].append(chunk.text)

class IntelligentPlanner:
    """Generates execution plans with context-aware resolution"""
    
    def __init__(self, 
                 chroma_collection: chromadb.Collection,  # Parameter name changed
                 param_resolver: ParamResolver,
                 llm_client: OpenAILLMClient,
                 dependency_manager: DependencyManager,
                 query_classifier: QueryClassifier):  # Added parameter
        self.collection = chroma_collection
        self.param_resolver = param_resolver
        self.llm_client = llm_client
        self.dependency_manager = dependency_manager
        self.query_classifier = query_classifier
        self.entity_registry = {}

    def generate_plan(self, query: str, entities: Dict, intents: Dict) -> Dict:
        """Generate validated plan with dependency tracking and enhanced fallbacks"""        

        query_type = intents.get('primary', {}).get('type', 'generic_search')
        entity_id_key = next((k for k in entities.keys() if k.endswith('_id')), None)
        
        try:
            # Determine primary entity type from resolved entities
            entity_type = next(
                (key.split('_')[0] for key in entities.keys() 
                if key.endswith('_id')), 
                'general'
            )
            
            # 1. Intent-aware initialization            
            context = {
                "query": query,
                "entities": json.dumps(entities, indent=2),
                "entity_id": f"${entity_id_key}" if entity_id_key else "",
                "entity_type": entity_id_key.split('_')[0] if entity_id_key else "item",
                "intents": intents,
                "resolved": json.dumps(self.entity_registry, indent=2),
                "query_type": self.query_classifier.classify(query)["primary_intent"],
                "template_hint": PROMPT_TEMPLATES.get(query_type, DEFAULT_TEMPLATE),               
            }

            # 2. LLM Planning with dependency hints
            raw_plan = self._llm_planning(
                prompt=PLAN_PROMPT.format(**context),
                dependencies=self.dependency_manager.graph
            )

            # 3. Plan validation and normalization
            validated_steps = PlanValidator(
                resolved_entities=entities
            ).validate_plan(raw_plan.get("plan", []))

            # 4. Dependency graph update
            self.dependency_manager.build_dependency_graph(validated_steps)

            # 5. Context-aware fallback generation
            if not validated_steps:
                validated_steps = FallbackHandler.generate_steps(
                    entities=entities,
                    intents=intents
                )

            # 6. Parameter specialization
            enhanced_plan = self._enhance_plan_with_context(
                validated_steps,
                query_type,
                entities
            )

            return {
                "plan": enhanced_plan,
                "metadata": {
                    "query_type": query_type,
                    "dependency_graph": self.dependency_manager.serialize()
                }
            }

        except Exception as e:
            # Graceful degradation with error context
            return FallbackHandler.generate_error_aware_plan(
                error=e,
                entities=entities,
                intents=intents
            )
    
    def _llm_planning(self, prompt: str, dependencies: nx.DiGraph) -> Dict:
        """Robust JSON parsing with validation"""
        response = self.llm_client.generate_response(prompt)
        
        try:
            # First try strict JSON parsing
            parsed = json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try JSON5 for more lenient parsing
                
                parsed = json5.loads(response)
            except:
                # Fallback to regex-based extraction
                json_str = re.search(r'\{.*\}', response, re.DOTALL)
                parsed = json.loads(json_str.group()) if json_str else {}
        
        # Validate root structure
        if not isinstance(parsed, dict):
            print(f"⚠️ LLM response is not a dictionary: {type(parsed)}")
            return {"plan": []}
        
        # Normalize plan key
        parsed["plan"] = parsed.get("plan") or parsed.get("steps") or []
        
        return parsed
    
    def _enhance_plan_with_context(
        self, 
        steps: List[Dict], 
        query_type: str, 
        entities: Dict
    ) -> List[Dict]:
        enhanced_steps = []
        
        for step in steps:
            if not isinstance(step, dict):
                continue

            endpoint_meta = self._get_endpoint_metadata(step["endpoint"])
            if not endpoint_meta:
                enhanced_steps.append(step)
                continue

            # Safely parse parameters metadata
            params_value = endpoint_meta.get("parameters", "[]")
            try:
                all_params = json.loads(params_value) if isinstance(params_value, str) else params_value
            except json.JSONDecodeError:
                all_params = []

            # Extract parameter types
            path_params = re.findall(r"{(\w+)}", endpoint_meta["path"])
            query_params = [
                p["name"] for p in all_params 
                if isinstance(p, dict) and p.get("in") == "query" and p.get("required", False)
            ]

            # Build parameter mapping
            resolved_params = {}
            
            # 1. Handle path parameters
            for param in path_params:
                if param in entities:
                    resolved_params[param] = f"${param}"
                else:
                    print(f"⚠️ Missing path parameter: {param}")

            # 2. Handle required query parameters
            for param in query_params:
                if param not in step.get("parameters", {}):
                    if param in entities:
                        resolved_params[param] = f"${param}"
                    else:
                        print(f"⚠️ Missing required query parameter: {param}")

            # 3. Merge parameters
            merged_params = {
                **step.get("parameters", {}),
                **resolved_params
            }

            enhanced_steps.append({
                **step,
                "endpoint": endpoint_meta["path"],
                "parameters": merged_params
            })
        
        return enhanced_steps
    
    def _get_endpoint_metadata(self, endpoint: str) -> Dict:
        """Retrieve parameter schema from ChromaDB with JSON deserialization"""
        results = self.collection.query(
            query_texts=[endpoint],
            n_results=1,
            include=["metadatas"]
        )
        
        if not results or not results["metadatas"][0]:
            return {}
        
        metadata = results["metadatas"][0][0]
        
        # Deserialize JSON fields
        if 'parameters' in metadata:
            try:
                metadata['parameters'] = json.loads(metadata['parameters'])
            except json.JSONDecodeError:
                metadata['parameters'] = []
        
        return metadata
    
class OpenAILLMClient:
    """Uses OpenAI LLM to generate execution plans dynamically."""

    def __init__(self, api_key, model="gpt-4-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_response(self, prompt):
        """Generates a response using OpenAI with logging."""
        print(f"📝 LLM Prompt:\n{prompt}\n")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}]
        )
        output = response.choices[0].message.content.strip()
        print(f"🤖 LLM Response:\n{output}\n")
        return output
