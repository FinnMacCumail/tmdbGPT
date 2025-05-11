# core/planner/question_routing.py

import json

QUESTION_TYPE_ROUTING = {
    "count": {
        "preferred_intents": ["credits.person", "credits.movie", "credits.tv"],
        "fallback_intents": ["details.person"],
        "response_format": "count_summary",
        "description": "Returns a numeric count of appearances"
    },
    "summary": {
        "preferred_intents": ["details.person", "details.movie", "details.tv"],
        "fallback_intents": [],
        "response_format": "summary",
        "description": "Returns a biography or synopsis"
    },
    "fact": {
        "preferred_intents": ["details.movie", "details.person"],
        "fallback_intents": [],
        "response_format": "summary",
        "description": "Provides a factual answer about a movie or person"
    },
    "timeline": {
        "preferred_intents": ["credits.person", "credits.tv", "credits.movie"],
        "fallback_intents": ["details.movie"],
        "response_format": "timeline",
        "description": "Returns entries ordered by release date"
    },
    "comparison": {
        "preferred_intents": ["details.movie", "details.person"],
        "fallback_intents": [],
        "response_format": "comparison",
        "description": "Returns a side-by-side comparison"
    },
    "list": {
        "preferred_intents": ["discovery.filtered", "discovery.advanced", "search.movie", "search.person", "credits.person"],
        "fallback_intents": [],
        "allowed_wildcards": ["discovery."],
        "response_format": "list",
        "description": "Returns a list of matching items"
    }
}


class SymbolicConstraintFilter:
    MEDIA_FILTER_ENTITIES = {"genre", "rating", "date",
                             "runtime", "votes", "language", "country"}

    @staticmethod
    def is_media_endpoint(produces_entities: list) -> bool:
        return any(media in produces_entities for media in {"movie", "tv"})

    @staticmethod
    def apply(matches: list, extraction_result: dict, resolved_entities: dict) -> list:
        question_type = extraction_result.get("question_type")
        query_intents = extraction_result.get("intents", [])
        resolved_keys = set(resolved_entities.keys())

        routing = QUESTION_TYPE_ROUTING.get(question_type, {})
        allowed_intents = set(routing.get(
            "preferred_intents", []) + routing.get("fallback_intents", []))

        filtered = []

        for match in matches:
            endpoint = match.get("endpoint") or match.get("path", "")
            metadata = match.get("metadata", match)
            supported_intents = SymbolicConstraintFilter._extract_supported_intents(
                metadata)
            consumes = SymbolicConstraintFilter._extract_consumed_entities(
                metadata)
            produces = SymbolicConstraintFilter._extract_produced_entities(
                metadata)

            missing_required_entities = []
            for key in resolved_keys:
                if key == "__query":
                    continue
                entity_type = SymbolicConstraintFilter._map_key_to_entity(key)
                if entity_type not in consumes:
                    if entity_type in SymbolicConstraintFilter.MEDIA_FILTER_ENTITIES and SymbolicConstraintFilter.is_media_endpoint(produces):
                        continue
                    missing_required_entities.append(key)

            soft_relaxed_fields = []
            if missing_required_entities:
                for key in missing_required_entities:
                    entity_type = SymbolicConstraintFilter._map_key_to_entity(
                        key)
                    if entity_type in SymbolicConstraintFilter.MEDIA_FILTER_ENTITIES:
                        soft_relaxed_fields.append(key)

                if soft_relaxed_fields and len(soft_relaxed_fields) == len(missing_required_entities):
                    match["soft_relaxed"] = soft_relaxed_fields
                    match["force_allow"] = True
                elif "credits" in endpoint and SymbolicConstraintFilter.is_media_endpoint(produces):
                    match["force_allow"] = True

            entity_penalty = 0.0
            if missing_required_entities and not SymbolicConstraintFilter.is_media_endpoint(produces):
                entity_penalty = 0.2 * len(missing_required_entities)

            qt_penalty = 0.0
            expected_patterns = SymbolicConstraintFilter._expected_patterns_for_question_type(
                question_type)
            if expected_patterns and not any(pat in endpoint.lower() for pat in expected_patterns):
                if not SymbolicConstraintFilter.is_media_endpoint(produces):
                    qt_penalty = 0.3

            existing_penalty = match.get("penalty", 0.0)
            match["penalty"] = existing_penalty + entity_penalty + qt_penalty

            intent_overlap = any(
                intent in allowed_intents for intent in supported_intents)
            wildcards = routing.get("allowed_wildcards", [])

            if not intent_overlap and wildcards:
                for wildcard in wildcards:
                    if any(supported_intent.startswith(wildcard) for supported_intent in supported_intents):
                        intent_overlap = True
                        break

            if intent_overlap or match.get("soft_relaxed") or match.get("force_allow"):
                filtered.append(match)

        return filtered

    @staticmethod
    def _map_key_to_entity(key: str) -> str:
        return key[:-3] if key.endswith("_id") else key

    @staticmethod
    def _extract_consumed_entities(metadata: dict) -> list:
        raw = metadata.get("consumes_entities", "[]")
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except:
            return []

    @staticmethod
    def _extract_produced_entities(metadata: dict) -> list:
        raw = metadata.get("produces_entities", "[]")
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except:
            return []

    @staticmethod
    def _extract_supported_intents(metadata: dict) -> list:
        raw = metadata.get("supported_intents", [])
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except:
                return []
        return raw

    @staticmethod
    def _expected_patterns_for_question_type(question_type: str) -> list:
        return {
            "count": ["/credits", "/details", "/movie_credits", "/tv_credits"],
            "summary": ["/details", "/person", "/movie", "/tv"],
            "timeline": ["/credits", "/discover"],
            "comparison": ["/details", "/credits"],
            "list": ["/discover", "/search", "/person", "/credits"],
            "fact": ["/details", "/movie", "/person"]
        }.get(question_type, [])
