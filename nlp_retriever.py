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

from fallback_handler import FallbackHandler
from llm_client import OpenAILLMClient


from param_resolver import ParamResolver
from llm_client import OpenAILLMClient
from dependency_manager import DependencyManager
from json import JSONDecodeError
from hybrid_retrieval_test import hybrid_search, convert_matches_to_execution_steps
# check redunacy here
#from param_utils import normalize_parameters


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

class ResponseFormatter:
    @staticmethod
    def format_responses(responses: list) -> list:
        """
        Format structured response entries with metadata or fall back to basic text.

        Args:
            responses (list): the raw response list from state.responses

        Returns:
            List[str]: formatted display-ready summaries
        """
        formatted = []
        for item in responses:
            if isinstance(item, str):
                tag = "📦 Fallback" if "trending" in item.lower() else "♻️ Relaxed" if "relaxed" in item.lower() else "📌"
                formatted.append(f"{tag} {item}")
            elif isinstance(item, dict):
                response_type = item.get("type")

                if response_type == "movie_summary":
                    title = item.get("title", "Unknown Title")
                    overview = item.get("overview", "")
                    source = item.get("source", "")
                    tag = "📦" if "trending" in source else "♻️" if "relaxed" in source else "🎬"
                    formatted.append(f"{tag} {title}: {overview}")
                    if source:
                        formatted.append(f"📦 Source: {source}")

                elif response_type == "person_profile":
                    name = item.get("title", "Unknown Person")
                    bio = item.get("overview", "")
                    formatted.append(f"👤 {name}: {bio}")

                elif "extracted" in item:
                    step = item.get("step", "")
                    for k, v in item["extracted"].items():
                        formatted.append(f"🧩 Extracted {k}: {v} (via {step})")

                else:
                    formatted.append(f"📎 {json.dumps(item)}")

        return formatted
    
def expand_plan_with_dependencies(state, newly_resolved: dict) -> list:
    """
    Use newly resolved entities to find and append follow-up steps to the plan.

    Args:
        state (AppState): current app state
        newly_resolved (dict): keys like {"person_id": 1234}

    Returns:
        List[dict]: list of new execution steps (if any)
    """
    if not newly_resolved:
        return []

    current_intents = state.extraction_result.get("intents", [])
    followup_matches = DependencyEndpointSuggester.suggest_followups(newly_resolved, current_intents)

    existing_endpoints = {step["endpoint"] for step in state.plan_steps}
    new_matches = [m for m in followup_matches if m["endpoint"] not in existing_endpoints]

    return convert_matches_to_execution_steps(new_matches, state.extraction_result, state.resolved_entities)

class DependencyEndpointSuggester:
    @staticmethod
    def suggest_followups(new_entities: dict, current_intents: list, limit: int = 10) -> list:
        """
        Given new resolved entities (e.g., person_id), suggest follow-up endpoints that are now actionable.

        Args:
            new_entities (dict): newly resolved keys like {'person_id': 6193}
            current_intents (list): list of active user intents from LLM extraction
            limit (int): max number of results to return

        Returns:
            list of dicts with suggested endpoint metadata
        """
        queries = []
        for entity_key in new_entities.keys():
            for intent in current_intents:
                prompt = f"Fetch endpoints requiring {entity_key} for intent {intent}"
                queries.append(prompt)

        all_matches = []
        for q in queries:
            results = hybrid_search(q, top_k=limit)
            all_matches.extend(results)

        # De-duplicate by endpoint
        seen = set()
        unique = []
        for m in all_matches:
            eid = m.get("endpoint")
            if eid and eid not in seen:
                seen.add(eid)
                unique.append(m)

        return unique

class PathRewriter:
    @staticmethod
    def rewrite(path: str, resolved_entities: dict) -> str:
        """
        Replaces unresolved placeholders in endpoint paths with resolved entity values.
        Example: /person/{person_id} → /person/123 if person_id is in resolved_entities.

        Args:
            path (str): the endpoint path with placeholders
            resolved_entities (dict): dictionary of resolved entity keys and values

        Returns:
            str: updated path with substitutions applied
        """

        def replacer(match):
            key = match.group(1)
            return str(resolved_entities.get(key, match.group(0)))

        return re.sub(r"{(\w+)}", replacer, path)

class PostStepUpdater:
    @staticmethod
    def update(state, step, json_data):
        path = step.get("endpoint")
        step_id = step.get("step_id", "unknown_step")

        extracted = {}

        if path.startswith("/search/"):
            print("🔎 Raw /search/person results:")
            for item in json_data.get("results", []):
                print(f"  → {item.get('name')} (id={item.get('id')})")
                if not isinstance(item, dict):
                    continue
                entity_id = item.get("id")
                entity_name = item.get("name") or item.get("title")
                if entity_id:
                    entity_type = PostStepUpdater._infer_entity_type(path)
                    if entity_type:
                        key = f"{entity_type}_id"
                        extracted.setdefault(key, []).append(entity_id)
                        print(f"🔍 Resolved {entity_type}: '{entity_name}' → {entity_id}")

        if extracted:
            state.resolved_entities.update(extracted)
            state.responses.append({"step": step_id, "extracted": extracted})

        return state

    @staticmethod
    def _infer_entity_type(path):
        if "person" in path:
            return "person"
        elif "movie" in path:
            return "movie"
        elif "tv" in path:
            return "tv"
        elif "company" in path:
            return "company"
        elif "collection" in path:
            return "collection"
        elif "keyword" in path:
            return "keyword"
        elif "network" in path:
            return "network"
        elif "genre" in path:
            return "genre"
        return None


class RerankPlanning:
    @staticmethod
    def rerank_matches(matches, resolved_entities):
        """
        Reorder and annotate matches based on parameter feasibility.
        Promote steps with resolved entities; demote those with missing params.
        """
        reranked = []
        for match in matches:
            endpoint = match["endpoint"]
            needs = []
            penalty = 0

            for key in ["person_id", "movie_id", "tv_id", "collection_id", "company_id"]:
                if f"{{{key}}}" in endpoint and not resolved_entities.get(key):
                    needs.append(key)
                    penalty += 0.4

            score = match["final_score"] - penalty
            match.update({
                "final_score": round(score, 3),
                "missing_entities": needs,
                "is_entrypoint": bool("/search" in endpoint)
            })
            reranked.append(match)

        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)

    @staticmethod
    def validate_parameters(endpoint, resolved_entities):
        """
        Check if endpoint has all the required resolved parameters.
        Return a flag indicating if the step is executable.
        """
        for key in ["person_id", "movie_id", "tv_id", "collection_id"]:
            if f"{{{key}}}" in endpoint and not resolved_entities.get(key):
                return False
        return True

    @staticmethod
    def filter_feasible_steps(ranked_matches, resolved_entities):
        """
        Return only steps that can be executed now, plus entrypoints.
        """
        feasible = []
        deferred = []
        for match in ranked_matches:
            if RerankPlanning.validate_parameters(match["endpoint"], resolved_entities):
                feasible.append(match)
            elif match.get("is_entrypoint"):
                feasible.append(match)
            else:
                deferred.append(match)

        return feasible, deferred

class ResultExtractor:
    @staticmethod
    def extract(endpoint: str, json_data: dict, resolved_entities: dict = None) -> list:
        summaries = []
        resolved_entities = resolved_entities or {}
        seen = set()

        print(f"📊 Top-level keys in response: {list(json_data.keys())}")
        for k, v in json_data.items():
            print(f"  → {k}: {type(v)}")

        # --- Special Case: /search/person
        if "/search/person" in endpoint:
            for result in json_data.get("results", []):
                name = result.get("name", "").strip()
                if name.lower() in seen:
                    continue
                seen.add(name.lower())

                known_for = result.get("known_for", [])
                known_titles = [
                    k.get("title") or k.get("name")
                    for k in known_for if isinstance(k, dict)
                    and (k.get("title") or k.get("name"))
                ]
                if not known_titles:
                    continue

                summaries.append({
                    "type": "movie_summary",
                    "title": name,
                    "overview": f"Known for: {', '.join(known_titles)}",
                    "source": endpoint
                })
            return summaries

        # --- General Case: Extract from list-based responses
        candidate_lists = [
            v for v in json_data.values() if isinstance(v, list)
        ]
        if not candidate_lists and "results" in json_data:
            candidate_lists = [json_data["results"]]

        for item_list in candidate_lists:
            for item in item_list:
                if not isinstance(item, dict):
                    continue

                title = item.get("title") or item.get("name", "")
                overview = (
                    item.get("overview")
                    or item.get("job")
                    or item.get("character")
                    or item.get("description")
                    or "No synopsis available."
                )

                if not title and not overview:
                    continue

                result_type = "keyword_summary" if "/keywords" in endpoint else "movie_summary"

                summaries.append({
                    "type": result_type,
                    "title": title,
                    "overview": str(overview),
                    "source": endpoint
                })

        # --- Flat dict fallback (for /person/{person_id} and others)
        flat_title = json_data.get("title") or json_data.get("name")
        flat_overview = json_data.get("overview") or json_data.get("biography") or ""

        if flat_title or flat_overview:
            is_person_profile = (
                "/person/" in endpoint
                and not any(k in endpoint for k in ["/credits", "/images", "/tv", "/movie"])
            )
            profile_type = "person_profile" if is_person_profile else "movie_summary"

            if is_person_profile:
                print(f"👤 Adding person_profile for {flat_title}")
            else:
                print(f"🎬 Adding movie_summary for {flat_title}")

            summaries.append({
                "type": profile_type,
                "title": flat_title,
                "overview": flat_overview.strip(),
                "source": endpoint
            })

        print(f"🎯 Endpoint for profile detection: {endpoint}")
        return summaries
    
class EnhancedIntentAnalyzer:
    def __init__(self, llm_client: OpenAILLMClient):
        self.nlp = spacy.load("en_core_web_trf")
        self.genre_map = self._fetch_genre_mappings()
        self.media_types = ["movie", "tv", "film", "show", "series"]
        self.classifier = HybridIntentClassifier(llm_client)
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

    def extract_entities(self, query: str) -> Dict:
        """Hybrid entity extraction pipeline"""    

        # Get LLM-enhanced classification
        intent_result = self.classifier.classify(query)

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

        # Merge results
        merged_entities = self._merge_entities(
            intent_result["entities"],
            entities
        )

        return dict(merged_entities)
    
    def _merge_entities(self, llm_entities: Dict[str, List[str]], spacy_entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge LLM-derived and spaCy-derived entities with deduplication"""
        merged = {}

        # Merge by entity type
        all_keys = set(llm_entities) | set(spacy_entities)
        for key in all_keys:
            llm_vals = set(llm_entities.get(key, []))
            spacy_vals = set(spacy_entities.get(key, []))
            merged[key] = list(llm_vals.union(spacy_vals))

        return merged
    
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
     
class HybridIntentClassifier:
    """Combines rule-based and LLM classification"""
    def __init__(self, llm_client: OpenAILLMClient):
        self.rule_classifier = QueryClassifier()
        self.llm = llm_client
        self.intent_prompt = """Classify the query intent and extract entities:
        Query: {query}
        
        Return JSON format:
        {{
            "primary_intent": "movie_details|tv_details|search|...",
            "entities": {{
                "entity_type": ["values"],
                "genre": ["action"],
                "person": ["Tom Hanks"]
            }}
        }}"""

    def classify(self, query: str) -> Dict:
        # First pass: Rule-based classification
        rule_result = self.rule_classifier.classify(query)
        
        # Use LLM fallback if generic intent or ambiguous
        if rule_result["primary_intent"] == "generic_search":
            return self._llm_classification(query)
        return rule_result

    def _llm_classification(self, query: str) -> Dict:
        prompt = self.intent_prompt.format(query=query)
        response = self.llm.generate_response(prompt)
        try:
            return json5.loads(response)
        except:
            return {"primary_intent": "generic_search", "entities": {}}
        
class SemanticAPIRetriever:
    """Retrieves relevant API endpoints using hybrid search"""
    def __init__(self, collection: chromadb.Collection):
        self.collection = collection

    def retrieve_context(self, query: str, intent: str, entities: Dict) -> str:
        print(f"\n🔍 RAG Retrieval Debug 🔍")
        print(f"| Query: {query}")
        print(f"| Intent: {intent}")
        print(f"| Entities: {entities}")
        
        try:
            # Build ChromaDB filter
            intent_filter = intent or "generic_search"
            where_clause = { "$or": [ {"intents": {"$eq": intent_filter}}, {"entity_types": {"$in": [list(entities.keys())[0]]}} ] }
            print(f"📦 ChromaDB Where Clause: {json.dumps(where_clause, indent=2)}")

            results = self.collection.query(
                query_texts=[f"Intent: {intent}\nQuery: {query}"],
                n_results=5,
                where=where_clause,
                include=["metadatas"]
            )
            
            print("📊 ChromaDB Results:")
            print(f"| Found {len(results['ids'][0])} endpoints")
            for i, (endpoint, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
                print(f"| Result {i+1}: {endpoint}")
                print(f"| Metadata: {json.dumps(metadata, indent=2)}")

            # Fallback if no results
            if not results['ids'][0]:
                print("⚠️ No endpoints found, using fallback context")
                return self._generate_fallback_context(entities)
                
            return self._format_context(results['metadatas'][0])
            
        except Exception as e:
            print(f"🔥 RAG Retrieval Error: {str(e)}")
            return ""
