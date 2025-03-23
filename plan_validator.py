# plan_validator.py
from typing import List, Dict

class PlanValidator:
    def __init__(self, resolved_entities: Dict):
        self.resolved_entities = resolved_entities

    def validate_plan(self, raw_plan: List[Dict]) -> List[Dict]:
        """Ensure valid parameters for person endpoint"""
        validated = []
        for step in raw_plan:
            step = self._fix_person_parameters(step)
            validated.append(step)
        return validated

    def _should_remove(self, step: Dict) -> bool:
        """Filter redundant entity resolution steps"""
        return (
            step.get('operation_type') == 'entity_resolution' and 
            self._entity_exists(step['endpoint'])
        )

    def _entity_exists(self, endpoint: str) -> bool:
        """Check if entity ID already resolved"""
        entity_type = endpoint.split('/')[-1]
        return f"{entity_type}_id" in self.resolved_entities

    def _add_step_metadata(self, step: Dict) -> Dict:
        """Enrich with execution requirements"""
        step['metadata'] = {
            'needs_entities': self._get_required_entities(step),
            'data_retrieval': not step.get('output_entities')
        }
        return step

    def _get_required_entities(self, step: Dict) -> List[str]:
        """Extract entity dependencies"""
        return [
            p[1:] for p in step.get('parameters', {}).values()
            if isinstance(p, str) and p.startswith("$")
        ]
    
    def _fix_person_parameters(self, step: Dict) -> Dict:
        """Force $person_id parameter format"""
        if "/person/" in step["endpoint"]:
            step["parameters"] = {"person_id": f"${self._get_person_id(step)}"}
        return step

    def _get_person_id(self, step: Dict) -> str:
        """Extract ID from endpoint or use resolved entity"""
        match = re.search(r"/person/(?:\{)?(\w+)(?:\})?", step["endpoint"])
        return match.group(1) if match else "person_id"