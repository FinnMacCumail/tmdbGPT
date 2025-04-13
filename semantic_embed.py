# semantic_embed.py (Fully Refactored with Search Handling, Normalized Scoring, and Dynamic Query Generation)
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

class SemanticEmbedder:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="tmdb_endpoints",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer(embedding_model)
        self.entity_hierarchy = self._define_entity_hierarchy()
        self.param_entity_map = self._build_parameter_entity_map()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.path_entity_map = {
            "/movie": "movie", "/tv": "tv", "/person": "person",
            "/company": "company", "/collection": "collection",
            "/network": "network", "/genre/movie": "movie",
            "/genre/tv": "tv", "/review": "review", "/credit": "credit",
            "/search/movie": "movie", "/search/tv": "tv", "/search/person": "person"
        }

    def _define_entity_hierarchy(self) -> Dict:
        return {
            "intents": {
                "recommendation": ["similarity", "suggested"],
                "discovery": ["filtered", "genre_based", "temporal", "advanced"],
                "search": ["basic", "multi"],
                "media_assets": ["image", "video"],
                "details": ["movie", "tv"],
                "credits": ["movie", "tv", "person"],
                "trending": ["popular", "top_rated"],
                "reviews": ["movie", "tv"],
                "collections": ["movie"],
                "companies": ["studio", "network"]
            },
            "entities": {
                "movie": ["genre", "year", "keyword", "image", "video", "review", "credit", "recommendation"],
                "tv": ["genre", "year", "image", "video", "review", "credit", "recommendation"],
                "person": ["movie", "tv", "image", "credit"],
                "company": ["image"],
                "network": ["image"],
                "collection": ["image"],
                "credit": ["movie", "tv"]
            }
        }

    def _build_parameter_entity_map(self) -> Dict:
        base_patterns = {
            r"person_id$": "person", r"movie_id$": "movie", r"tv_id$": "tv",
            r"credit_id$": "credit", r"company_id$": "company", r"network_id$": "network",
            r"collection_id$": "collection", r"season_number$": "tv.season",
            r"episode_number$": "tv.episode", r"with_genres$": "genre",
            r"with_keywords$": "keyword", r"with_companies$": "company",
            r"year$": "date", r"primary_release_year$": "date",
            r"first_air_date_year$": "date", r"region$": "country", r"language$": "language",
            r"vote_average.*": "rating", r"primary_release_date.*": "date", r"release_date.*": "date",
            r"with_people$": "person", r"query$": "keyword", r"name$": "keyword"
        }
        for parent, children in self.entity_hierarchy["entities"].items():
            for child in children:
                base_patterns[f"{parent}.{child}"] = f"{parent}.{child}"
        return base_patterns

    def _parse_llm_response(self, response) -> List[str]:
        try:
            content = response.choices[0].message.content
            lines = content.strip().split("\n")
            cleaned = []
            for line in lines:
                if ". " in line:
                    cleaned.append(line.split(". ", 1)[1])
                else:
                    cleaned.append(line.strip())
            return cleaned
        except Exception as e:
            print(f"⚠️ LLM Response Parse Error: {str(e)}")
            return []

    def _fallback_template_queries(self, param_names: List[str]) -> List[str]:
        fallback = []
        if any("person" in p for p in param_names):
            fallback.append("Search for movies starring a specific actor.")
        if any("genre" in p for p in param_names):
            fallback.append("List action movies released in 2020.")
        if any("keyword" in p for p in param_names):
            fallback.append("Find movies with the keyword 'space'.")
        return fallback

    def _generate_query_examples(self, endpoint: str, details: Dict) -> List[str]:
        param_names = [p.get("name", "") for p in details.get("parameters", [])]
        fallback_queries = []

        if any("person" in p for p in param_names):
            fallback_queries.append("Search for movies starring Brad Pitt.")
        if any("genre" in p for p in param_names):
            fallback_queries.append("List action movies released in 2020.")
        if any("vote_average" in p for p in param_names):
            fallback_queries.append("Show movies with rating above 8.")

        # Boost /search/* endpoints
        if endpoint.startswith("/search/") or any(p in ["query", "name"] for p in param_names):
            fallback_queries.append("Find a movie called Inception.")

        prompt = f"""
        You are a TMDB assistant.

        Endpoint: {endpoint}
        Description: {details.get('description', '')}
        Parameters: {json.dumps(details.get('parameters', []), indent=2)}

        Generate 2 example user queries that match the purpose of this endpoint.
        Return a numbered list only.
        """
        try:
            response = self.openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You generate helpful search queries for TMDB endpoints."},
                    {"role": "user", "content": prompt.strip()}
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content
            return self._parse_llm_response(response)
        except Exception as e:
            print(f"⚠️ LLM Query Generation Failed: {str(e)}")
            return fallback_queries

    def _detect_intents(self, endpoint: str, params: List[Dict]) -> List[Dict]:
        intent_scores = defaultdict(float)

        exact_patterns = {
            r"^/movie/{movie_id}$": ["details.movie"],
            r"^/tv/{tv_id}$": ["details.tv"],
            r"^/person/{person_id}$": ["credits.person"],
            r"^/movie/{movie_id}/recommendations$": ["recommendation.similarity"],
            r"^/movie/{movie_id}/similar$": ["recommendation.similarity"],
            r"^/tv/{tv_id}/recommendations$": ["recommendation.similarity"],
            r"^/tv/{tv_id}/similar$": ["recommendation.similarity"],
            r"^/movie/popular$": ["trending.popular"],
            r"^/movie/top_rated$": ["trending.popular"],
            r"^/movie/upcoming$": ["discovery.temporal"],
            r"^/tv/popular$": ["trending.popular"],
            r"^/tv/top_rated$": ["trending.popular"],
            r"^/tv/on_the_air$": ["discovery.temporal"],
            r"^/tv/airing_today$": ["discovery.temporal"],
            r"^/movie/{movie_id}/images$": ["media_assets.image"],
            r"^/tv/{tv_id}/images$": ["media_assets.image"],
            r"^/person/{person_id}/images$": ["media_assets.image"],
            r"^/tv/{tv_id}/reviews$": ["reviews"],
            r"^/movie/{movie_id}/reviews$": ["reviews"]
        }

        fuzzy_patterns = {
            r"/search": ["search.multi"],
            r"/images": ["media_assets.image"],
            r"/videos": ["media_assets.video"],
            r"/credits": ["credits"],
            r"/reviews": ["reviews"],
            r"/trending|/popular|/top_rated": ["trending.popular"],
            r"/latest|/upcoming|/now_playing|/on_the_air": ["discovery.temporal"],
            r"/collection": ["collections.movie"],
            r"/company": ["companies.studio"],
            r"/network": ["companies.network"]
        }

        # Match exact endpoint patterns
        for pattern, intents in exact_patterns.items():
            if re.fullmatch(pattern, endpoint):
                for intent in intents:
                    intent_scores[intent] += 1.0

        # Match fallback patterns by partial inclusion
        for pattern, intents in fuzzy_patterns.items():
            if re.search(pattern, endpoint):
                for intent in intents:
                    intent_scores[intent] += 0.3

        # Parameter-based intent boosts
        for param in params:
            pname = param.get("name", "").lower()
            etype = param.get("entity_type", "")

            if "with_genres" in pname or etype == "genre":
                intent_scores["discovery.genre_based"] += 0.6
            if "vote_average.gte" in pname or "rating" in pname:
                intent_scores["discovery.filtered"] += 0.6
            if "with_people" in pname or etype == "person":
                intent_scores["credits.person"] += 0.6
            if etype == "date" or "year" in pname:
                intent_scores["discovery.temporal"] += 0.6
            if "certification" in pname:
                intent_scores["discovery.filtered"] += 0.5
            if "sort_by" in pname:
                intent_scores["discovery.filtered"] += 0.4
            if "region" in pname or etype == "country":
                intent_scores["regional"] += 0.5
            if "keyword" in pname:
                intent_scores["search.multi"] += 0.5

        # Additional endpoint context boosts
        if "/search/" in endpoint:
            intent_scores["search.multi"] += 0.4
        if "/discover/" in endpoint:
            intent_scores["discovery.filtered"] += 0.5
        if "/similar" in endpoint or "/recommendations" in endpoint:
            intent_scores["recommendation.similarity"] += 0.6
        if "/reviews" in endpoint:
            intent_scores["reviews"] += 0.6

        if not intent_scores:
            return [{"intent": "general", "confidence": 0.3}]

        max_score = max(intent_scores.values())
        return [
            {"intent": k, "confidence": round(v / max_score, 2)}
            for k, v in sorted(intent_scores.items(), key=lambda x: -x[1])
            if v >= 0.3
        ]

    def _map_param_to_entity(self, pname: str) -> str:
        for pattern, entity in self.param_entity_map.items():
            if re.search(pattern, pname):
                return entity
        return "general"

    def _fallback_entities_with_confidence(self, endpoint: str) -> Dict[str, float]:
        scores = defaultdict(float)
        for fragment, entity in self.path_entity_map.items():
            if fragment in endpoint:
                score = len(fragment.split("/")) / 10
                scores[entity] = max(scores[entity], round(score, 2))
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def _detect_entities(self, params: List[Dict], endpoint: str) -> List[str]:
        entities = set()
        for param in params:
            entity = self._map_param_to_entity(param.get("name", ""))
            if entity != "general":
                entities.add(entity.split(".")[0])

        path_params = re.findall(r"{(\w+)}", endpoint)
        for p in path_params:
            entity = self._map_param_to_entity(p)
            if entity != "general":
                entities.add(entity.split(".")[0])

        fallback_scores = self._fallback_entities_with_confidence(endpoint)
        for entity, score in fallback_scores.items():
            if score > 0.2:
                entities.add(entity)

        return list(entities)

    def _create_embedding_text(self, endpoint: str, details: Dict) -> str:
        examples = self._generate_query_examples(endpoint, details)
        intents = self._detect_intents(endpoint, details.get("parameters", []))
        entities = self._detect_entities(details.get("parameters", []), endpoint)
        top_intents = sorted(intents, key=lambda i: -i["confidence"])[:3]
        return "\n".join([
            f"Endpoint: {endpoint}",
            f"Description: {details.get('description', '')[:250]}...",
            f"Key Intents: {', '.join(f'{i['intent']} (conf: {i['confidence']})' for i in top_intents)}",
            f"Key Entities: {', '.join(entities)}",
            "Example Queries:",
            *[f"- {q}" for q in examples[:3]]
        ])

    def _create_metadata(self, endpoint: str, details: Dict) -> Dict:
        params = details.get("parameters", [])
        return {
            "path": endpoint,
            "intents": json.dumps(self._detect_intents(endpoint, params)),
            "entities": ", ".join(self._detect_entities(params, endpoint)),
            "parameters": json.dumps(params)
        }

    def process_endpoints(self, spec_path: str = "data/tmdb.json"):
        with open(spec_path) as f:
            spec = json.load(f)

        ids, metadatas, embeddings = [], [], []
        for endpoint, details in spec["paths"].items():
            verb = details.get("get")
            if not verb:
                continue
            metadata = self._create_metadata(endpoint, verb)
            embed_text = self._create_embedding_text(endpoint, verb)
            vector = self.embedder.encode(embed_text).tolist()
            ids.append(endpoint)
            metadatas.append(metadata)
            embeddings.append(vector)

        self.collection.upsert(ids=ids, metadatas=metadatas, embeddings=embeddings)

if __name__ == "__main__":
    SemanticEmbedder().process_endpoints()
