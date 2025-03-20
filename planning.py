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