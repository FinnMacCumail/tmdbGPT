# fallback_handler.py
from typing import Dict, List

class FallbackHandler:
    @staticmethod
    def generate_direct_access(entities: Dict) -> List[Dict]:
        """Create valid direct access steps"""
        steps = []
        for entity_key, entity_id in entities.items():
            if "_id" in entity_key:
                entity_type = entity_key.split("_")[0]
                steps.append({
                    "step_id": f"direct_{entity_type}",
                    "description": f"Direct {entity_type} access",
                    "endpoint": f"/{entity_type}/{{{entity_key}}}",
                    "method": "GET",
                    "parameters": {entity_key: f"${entity_key}"},
                    "operation_type": "data_retrieval"
                })
        return steps

    @staticmethod
    def format_fallback(entities: Dict) -> str:
        """Create response from raw entities"""
        return "\n".join(
            f"{k.replace('_id', '').title()}: {v}"
            for k, v in entities.items()
        )