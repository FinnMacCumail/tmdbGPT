# fallback_handler.py
from typing import Dict, List
from dependency_manager import ExecutionState

class FallbackHandler:
    @staticmethod
    def generate_steps(entities: Dict, intents: Dict) -> List[Dict]:  # Remove query_type
        """Create fallback steps based on available entities"""
        steps = []
        
        # Entity priority: person > movie > tv > genre
        if 'person_id' in entities:
            steps.append({
                "step_id": "fallback_person",
                "endpoint": f"/person/{entities['person_id']}",
                "method": "GET"
            })
        elif 'movie_id' in entities:
            steps.append({
                "step_id": "fallback_movie",
                "endpoint": f"/movie/{entities['movie_id']}",
                "method": "GET"
            })
        else:
            steps.append({
                "step_id": "fallback_discover",
                "endpoint": "/discover/movie",
                "method": "GET",
                "parameters": {
                    "sort_by": "popularity.desc",
                    "page": 1
                }
            })
        return steps

    @staticmethod
    def format_fallback(entities: Dict) -> str:
        """Create response from raw entities"""
        return "\n".join(
            f"{k.replace('_id', '').title()}: {v}"
            for k, v in entities.items()
        )
    
class AdaptiveFallback:
    def generate_fallback(self, state: ExecutionState) -> List[Dict]:
        """Generate steps based on available entities and intents"""
        # Prioritize entity-based fallbacks
        for entity in ['person', 'movie', 'tv']:
            if f"{entity}_id" in state.resolved_entities:
                return [{
                    "step_id": f"fallback_{entity}",
                    "endpoint": f"/{entity}/{state.resolved_entities[f'{entity}_id']}",
                    "operation_type": "data_retrieval",
                    "requires_entities": [f"{entity}_id"]
                }]
        
        # Intent-based fallback
        if state.detected_intents.get('primary') == 'trending':
            return [{
                "step_id": "fallback_trending",
                "endpoint": "/trending/movie/week",
                "operation_type": "data_retrieval",
                "parameters": {"time_window": "week"}
            }]
        
        # Default discovery fallback
        return [{
            "step_id": "fallback_discover",
            "endpoint": "/discover/movie",
            "operation_type": "data_retrieval",
            "parameters": {"sort_by": "popularity.desc"}
        }]