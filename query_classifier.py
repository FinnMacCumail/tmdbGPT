import re
from typing import Dict

class QueryClassifier:
    INTENT_MAP = {
    # ================== CORE ENTITY DETAILS ==================
    "movie_details": {
        "patterns": [
            r"\b(movie|film)\b.*\b(details|info|information|about)\b",
            r"\b(show me|tell me|what is) (the )?.* (movie|film)\b"
        ],
        "endpoints": [
            "/movie/{movie_id}",
            "/movie/{movie_id}/release_dates",
            "/movie/{movie_id}/keywords",
            "/movie/{movie_id}/videos",
            "/movie/{movie_id}/translations",
            "/movie/{movie_id}/external_ids"
        ],
        "priority": 1
    },

    "tv_series_details": {
        "patterns": [
            r"\b(tv show|series|episode)\b.*\b(details|info)\b",
            r"\b(about|information).* (tv series|show)\b"
        ],
        "endpoints": [
            "/tv/{tv_id}",
            "/tv/{tv_id}/content_ratings",
            "/tv/{tv_id}/keywords",
            "/tv/{tv_id}/translations",
            "/tv/{tv_id}/external_ids"
        ],
        "priority": 2
    },

    # ================== MEDIA ASSETS ==================
    "image_assets": {
        "patterns": [
            r"\b(images|posters|photos|stills)\b.*\b(from|of)\b",
            r"\bmovie stills (from|of)\b"
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
            r"\b(trailers|clips|teasers)\b.*\b(movie|film|show)\b",
            r"\b(show me|play) (the )?.* (trailer|clip)\b"
        ],
        "endpoints": [
            "/movie/{movie_id}/videos",
            "/tv/{tv_id}/videos"
        ],
        "priority": 4
    },

    # ================== CREDITS & RELATIONSHIPS ==================
    "credits_metadata": {
        "patterns": [
            r"\b(cast|crew|actors|directors)\b.*\b(movie|film|show)\b",
            r"\b(who is|who are) (in|starring)\b"
        ],
        "endpoints": [
            "/movie/{movie_id}/credits",
            "/tv/{tv_id}/credits",
            "/person/{person_id}/combined_credits",
            "/tv/{tv_id}/season/{season_number}/credits",
            "/tv/{tv_id}/season/{season_number}/episode/{episode_number}/credits"
        ],
        "priority": 5
    },

    "recommendations": {
        "patterns": [
            r"\b(similar|recommended|related|like this)\b.*\b(movie|show)\b",
            r"\b(if you like|you might enjoy)\b"
        ],
        "endpoints": [
            "/movie/{movie_id}/recommendations",
            "/tv/{tv_id}/recommendations",
            "/movie/{movie_id}/similar",
            "/tv/{tv_id}/similar"
        ],
        "priority": 6
    },

    # ================== SEARCH & DISCOVERY ==================
    "advanced_discovery": {
        "patterns": [
            r"\b(filter|discover|find).*\bby (genre|year|rating)\b",
            r"\b(movies|shows) (with|featuring)\b.*\b(from|during)\b"
        ],
        "endpoints": [
            "/discover/movie",
            "/discover/tv"
        ],
        "priority": 7
    },

    "multi_search": {
        "patterns": [
            r"\b(search|find)\b.*\b(movies?|shows?|people)\b",
            r"\blooking for.*(called|named)\b"
        ],
        "endpoints": [
            "/search/movie",
            "/search/tv",
            "/search/person",
            "/search/company",
            "/search/collection",
            "/person/popular"
        ],
        "priority": 8
    },

    # ================== TEMPORAL CONTENT ==================
    "trending_content": {
        "patterns": [
            r"\b(trending|popular|top rated|what's hot)\b",
            r"\b(currently|now) (popular|trending)\b"
        ],
        "endpoints": [
            "/trending/{media_type}/{time_window}",
            "/movie/popular",
            "/tv/popular",
            "/movie/now_playing",
            "/tv/airing_today",
            "/movie/top_rated",
            "/tv/top_rated",
            "/movie/upcoming",
            "/tv/on_the_air"
        ],
        "priority": 9
    },

    "historical_content": {
        "patterns": [
            r"\b(from|during|in) (the )?\d{4}s\b",
            r"\b(old|classic|vintage) (movies|films|shows)\b"
        ],
        "endpoints": [
            "/discover/movie?sort_by=release_date.asc",
            "/discover/tv?sort_by=first_air_date.asc"
        ],
        "priority": 10
    },

    # ================== SYSTEM METADATA ==================
    "genre_metadata": {
        "patterns": [
            r"\b(genres|categories|types) of (movies|shows)\b",
            r"\bwhat (genre|category)\b"
        ],
        "endpoints": [
            "/genre/movie/list",
            "/genre/tv/list"
        ],
        "priority": 11
    },

    "certification_metadata": {
        "patterns": [
            r"\b(rated|certified|rating)\b.*\b(R|PG-13|TV-MA)\b",
            r"\bmaturity ratings for\b"
        ],
        "endpoints": [
            "/certification/movie/list",
            "/certification/tv/list"
        ],
        "priority": 12
    },

    "watch_providers": {
        "patterns": [
            r"\b(stream|watch).*\b(on|via)\b",
            r"\b(available|where to watch)\b"
        ],
        "endpoints": [
            "/watch/providers/movie",
            "/watch/providers/tv"
        ],
        "priority": 13
    },

    # ================== SUPPORTING ENTITIES ==================
    "company_operations": {
        "patterns": [
            r"\b(production companies|studios)\b.*\b(movie|film|show)\b",
            r"\bmade by (studio|company)\b"
        ],
        "endpoints": [
            "/company/{company_id}",
            "/search/company",
            "/company/{company_id}/images"
        ],
        "priority": 14
    },

    "network_operations": {
        "patterns": [
            r"\b(tv networks|channels)\b.*\b(showing|airing)\b",
            r"\b(original network|broadcast by)\b"
        ],
        "endpoints": [
            "/network/{network_id}",
            "/search/network",
            "/network/{network_id}/images"
        ],
        "priority": 15
    },

    # ================== USER CONTENT ==================
    "review_analysis": {
        "patterns": [
            r"\b(reviews|ratings|critic opinions)\b.*\b(movie|show)\b",
            r"\bwhat (critics|reviewers) say\b"
        ],
        "endpoints": [
            "/movie/{movie_id}/reviews",
            "/tv/{tv_id}/reviews",
            "/review/{review_id}"
        ],
        "priority": 16
    },

    "list_operations": {
        "patterns": [
            r"\b(collections|movie lists|curated)\b",
            r"\b(set|group) of (movies|films)\b"
        ],
        "endpoints": [
            "/collection/{collection_id}",
            "/search/collection"
        ],
        "priority": 17
    },

    # ================== PERSON CREDITS ==================
    "person_credits": {
        "patterns": [
            r"\b(credits|appearances|roles)\b.*\b(person|actor|actress|cast)\b",
            r"\bmovies? (featuring|by)\b.*"
        ],
        "endpoints": [
            "/person/{person_id}/movie_credits",
            "/person/{person_id}/tv_credits",
            "/person/{person_id}/combined_credits"
        ],
        "priority": 18
    },

    # ================== FALLBACK METADATA ==================
    "basic_details": {
        "patterns": [
            r"\b(details|info|data|metadata)\b.*",
            r"\bget information on\b"
        ],
        "endpoints": [
            "/credit/{credit_id}",
            "/person/{person_id}",
            "/tv/{tv_id}/season/{season_number}",
            "/tv/{tv_id}/season/{season_number}/episode/{episode_number}"
        ],
        "priority": 19
    }
}

    PRIORITY_ORDER = [
        "image_assets", "video_assets",
        "movie_details", "tv_series_details",
        "credits_metadata", "recommendations",
        "advanced_discovery", "multi_search",
        "trending_content", "historical_content",
        "genre_metadata", "certification_metadata",
        "watch_providers", "company_operations",
        "network_operations", "review_analysis",
        "list_operations"
    ]

    def classify(self, query: str) -> Dict:
        """Dynamically classify query intent using regex patterns"""
        query = query.lower()
        matched_intents = []

        for intent, patterns in self.INTENT_PATTERNS.items():
            if any(re.search(pattern, query) for pattern in patterns):
                matched_intents.append(intent)

        return {
            "primary_intent": matched_intents[0] if matched_intents else "generic_search",
            "secondary_intents": matched_intents[1:]
        }