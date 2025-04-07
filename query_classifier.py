import re
import json
from typing import Dict
from openai import OpenAI

class QueryClassifier:
    INTENT_MAP = {
        # ================== CORE MEDIA DETAILS ==================
        "movie_details": {
            "patterns": [
                r"\b(movie|film)\b.*\b(detail|info|data|about)\b",
                r"\b(show me|tell me|what is) (the )?.* (movie|film)\b"
            ],
            "endpoints": [
                "/movie/{movie_id}",
                "/movie/{movie_id}/release_dates",
                "/movie/{movie_id}/keywords",
                "/movie/{movie_id}/translations",
                "/movie/{movie_id}/external_ids"
            ],
            "priority": 1
        },

        "tv_details": {
            "patterns": [
                r"\b(tv show|series|episode)\b.*\b(detail|info)\b",
                r"\b(about|information).* (tv series|show)\b"
            ],
            "endpoints": [
                "/tv/{tv_id}",
                "/tv/{tv_id}/content_ratings",
                "/tv/{tv_id}/keywords",
                "/tv/{tv_id}/translations",
                "/tv/{tv_id}/external_ids",
                "/tv/{tv_id}/season/{season_number}",
                "/tv/{tv_id}/season/{season_number}/episode/{episode_number}"
            ],
            "priority": 2
        },

        # ================== MEDIA ASSETS ==================
        "image_assets": {
            "patterns": [
                r"\b(image|poster|photo|still|logo)\b",
                r"\bmovie stills?\b",
                r"\b(tv|movie) (posters?|logos?)\b"
            ],
            "endpoints": [
                "/movie/{movie_id}/images",
                "/tv/{tv_id}/images",
                "/person/{person_id}/images",
                "/collection/{collection_id}/images",
                "/company/{company_id}/images",
                "/network/{network_id}/images",
                "/tv/{tv_id}/season/{season_number}/images",
                "/tv/{tv_id}/season/{season_number}/episode/{episode_number}/images"
            ],
            "priority": 3
        },

        "video_assets": {
            "patterns": [
                r"\b(trailer|clip|teaser|video)\b",
                r"\b(show me|play) (the )?.* (trailer|clip)\b"
            ],
            "endpoints": [
                "/movie/{movie_id}/videos",
                "/tv/{tv_id}/videos"
            ],
            "priority": 4
        },

        # ================== CREDITS & PEOPLE ==================
        "credit_metadata": {
            "patterns": [
                r"\b(cast|crew|actor|director|producer)\b",
                r"\b(who (is|are) (in|starring)\b)",
                r"\b(credits?|roles?)\b"
            ],
            "endpoints": [
                "/movie/{movie_id}/credits",
                "/tv/{tv_id}/credits",
                "/person/{person_id}/combined_credits",
                "/tv/{tv_id}/season/{season_number}/credits",
                "/tv/{tv_id}/season/{season_number}/episode/{episode_number}/credits",
                "/credit/{credit_id}"
            ],
            "priority": 5
        },

        # ================== DISCOVERY & SEARCH ==================
        "advanced_discovery": {
            "patterns": [
                r"\bdiscover\b.*\b(movies?|shows?)\b",
                r"\bfilter\b.*\b(genre|year|rating|company)\b",
                r"\b(find|search).*\bby (actor|director|studio)\b"
            ],
            "endpoints": [
                "/discover/movie",
                "/discover/tv"
            ],
            "priority": 6
        },

        "multi_search": {
            "patterns": [
                r"\bsearch\b.*\b(movies?|shows?|people|companies)\b",
                r"\blooking for.*(called|named|titled)\b"
            ],
            "endpoints": [
                "/search/movie",
                "/search/tv",
                "/search/person",
                "/search/company",
                "/search/collection"
            ],
            "priority": 7
        },

        # ================== TRENDING & POPULARITY ==================
        "trending_content": {
            "patterns": [
                r"\b(trending|popular|top rated|what's hot)\b",
                r"\b(currently|now) (popular|trending)\b",
                r"\b(hot|popular) (right now|this week)\b"
            ],
            "endpoints": [
                "/trending/{media_type}/{time_window}",
                "/movie/popular",
                "/tv/popular",
                "/movie/top_rated",
                "/tv/top_rated",
                "/movie/upcoming",
                "/movie/now_playing",
                "/tv/airing_today",
                "/tv/on_the_air",
                "/person/popular"
            ],
            "priority": 8
        },

        # ================== TEMPORAL CONTENT ==================
        "temporal_content": {
            "patterns": [
                r"\b(newest|latest|recently added)\b",
                r"\b(upcoming|coming soon)\b",
                r"\b(just added|released today)\b"
            ],
            "endpoints": [
                "/movie/latest",
                "/tv/latest",
                "/movie/upcoming",
                "/tv/on_the_air"
            ],
            "priority": 9
        },

        # ================== GENRES & CLASSIFICATIONS ==================
        "genre_metadata": {
            "patterns": [
                r"\b(genres?|categories?)\b",
                r"\btypes? of (movies|shows)\b"
            ],
            "endpoints": [
                "/genre/movie/list",
                "/genre/tv/list"
            ],
            "priority": 10
        },

        # ================== COMPANIES & NETWORKS ==================
        "company_operations": {
            "patterns": [
                r"\b(studio|production company|distributor)\b",
                r"\bmade by (studio|company)\b"
            ],
            "endpoints": [
                "/company/{company_id}",
                "/search/company",
                "/company/{company_id}/images"
            ],
            "priority": 11
        },

        "network_operations": {
            "patterns": [
                r"\b(tv network|channel|broadcaster)\b",
                r"\b(original network|aired on)\b"
            ],
            "endpoints": [
                "/network/{network_id}",
                "/search/network",
                "/network/{network_id}/images"
            ],
            "priority": 12
        },

        # ================== REVIEWS & RATINGS ==================
        "review_analysis": {
            "patterns": [
                r"\b(reviews?|ratings?|critic opinions?)\b",
                r"\bwhat (critics|reviewers) say\b"
            ],
            "endpoints": [
                "/movie/{movie_id}/reviews",
                "/tv/{tv_id}/reviews",
                "/review/{review_id}"
            ],
            "priority": 13
        },

        # ================== COLLECTIONS & SERIES ==================
        "collection_operations": {
            "patterns": [
                r"\b(collections?|series|franchise)\b",
                r"\bmovie series\b"
            ],
            "endpoints": [
                "/collection/{collection_id}",
                "/search/collection",
                "/collection/{collection_id}/images"
            ],
            "priority": 14
        },

        # ================== RECOMMENDATIONS ==================
        "recommendation_engine": {
            "patterns": [
                r"\b(similar|recommended|related|like this)\b",
                r"\bif you (like|enjoy)\b"
            ],
            "endpoints": [
                "/movie/{movie_id}/recommendations",
                "/tv/{tv_id}/recommendations",
                "/movie/{movie_id}/similar",
                "/tv/{tv_id}/similar"
            ],
            "priority": 15
        }
    }

    PRIORITY_ORDER = [
        "image_assets", "video_assets",
        "movie_details", "tv_details",
        "credit_metadata", "recommendation_engine",
        "advanced_discovery", "multi_search",
        "trending_content", "temporal_content",
        "genre_metadata", "company_operations",
        "network_operations", "review_analysis",
        "collection_operations"
    ]
    
    def __init__(self, api_key: str = None):
        self.llm_client = OpenAI(api_key=api_key) if api_key else None
        self.intent_labels = [
            "biographical", "discovery", "comparative", 
            "temporal", "multimedia", "credits",
            "recommendation", "popularity", "combined"
        ]

    def classify(self, query: str) -> Dict:
        """Hybrid classification with LLM fallback"""
        rule_result = self._rule_based_classification(query)
        
        if rule_result["primary_intent"] == "generic_search" and self.llm_client:
            return self._llm_classification(query)
            
        return rule_result

    def _llm_classification(self, query: str) -> Dict:
        """LLM-based intent classification"""
        prompt = f"""Analyze this media query and return JSON with:
        - primary_intent (from {self.intent_labels})
        - secondary_intents
        - implied_entities
        
        Query: {query}"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ LLM Classification Error: {str(e)}")
            return self._rule_based_classification(query)

    def _rule_based_classification(self, query: str) -> Dict:
        """Original regex-based classification"""
        matched = []
        query = query.lower()
        
        for intent, config in self.INTENT_MAP.items():
            if any(re.search(p, query) for p in config["patterns"]):
                matched.append(intent)
        
        return {
            "primary_intent": matched[0] if matched else "generic_search",
            "secondary_intents": matched[1:] if matched else [],
            "implied_entities": []
        }