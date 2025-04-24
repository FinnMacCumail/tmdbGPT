# fallback_handler.py
from typing import Dict, List

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
                },
                "fallback_injected": True  # ✅ Add this flag
            })
        return steps            