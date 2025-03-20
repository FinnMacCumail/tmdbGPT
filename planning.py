from enum import Enum, auto
from typing import Dict

class StepType(Enum):
    ENTITY_DISCOVERY = auto()
    DATA_RETRIEVAL = auto()
    RELATIONSHIP_MAPPING = auto()

def classify_step(step: Dict) -> StepType:
    if step.get('output_entities'):
        return StepType.ENTITY_DISCOVERY
    if '/search/' in step.get('endpoint', ''):
        return StepType.ENTITY_DISCOVERY
    if any(key in step.get('parameters', {}) for key in ['query', 'filter']):
        return StepType.DATA_RETRIEVAL
    return StepType.RELATIONSHIP_MAPPING

def _validate_single_step(raw_step: Dict) -> Optional[Dict]:
    # Existing validation...
    
    # State-aware validation
    missing_deps = [
        dep for dep in raw_step.get("depends_on", [])
        if dep not in state["step_status"]
    ]
    if missing_deps:
        print(f"Missing step dependencies: {missing_deps}")
        return None
    
def _should_skip_step(raw_step: Dict, state: ExecutionState) -> bool:
    """State-aware step validation with data step awareness"""
    step_type = classify_step(raw_step)
    
    # Never skip data retrieval steps
    if step_type == StepType.DATA_RETRIEVAL:
        return False
        
    # Original entity-based skipping logic
    return step_outputs.issubset(existing_entities)