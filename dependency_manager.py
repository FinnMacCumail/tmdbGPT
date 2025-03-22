# dependency_manager.py
from typing import Dict, List, Set, TypedDict, Any, Optional
import re
from pydantic import BaseModel, ConfigDict
from networkx import DiGraph
import uuid
import time

class EntityLifecycleEntry(TypedDict):
    type: str  # 'production' or 'consumption'
    step_id: str
    timestamp: float

class ExecutionState(BaseModel):
    """Central state container for workflow execution"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Add error tracking field
    error: Optional[str] = None
    
    # Entity Handling
    raw_entities: Dict[str, Any] = {}  
    resolved_entities: dict[str, Any] = {}

    # Execution Tracking
    detected_intents: Dict[str, Any] = {}
    data_registry: dict[str, Any] = {}
    pending_steps: list[dict] = []
    completed_steps: list[dict] = []

    # System Internals
    entity_lifecycle: dict[str, list[EntityLifecycleEntry]] = {}
    dependency_graph: DiGraph = DiGraph()

    def track_entity_activity(self, entity: str, activity_type: str, step: Dict):
        """Record entity production or consumption"""
        entry = EntityLifecycleEntry(
            type=activity_type,
            step_id=step.get('step_id', 'unknown'),
            timestamp=time.time()
        )
        self.entity_lifecycle.setdefault(entity, []).append(entry)

    def get_entity_dependencies(self, entity: str) -> List[str]:
        """Get steps that depend on a particular entity"""
        return [edge[1] for edge in self.dependency_graph.edges if edge[0] == entity]

class DependencyManager:
    def __init__(self):
        self.execution_state = ExecutionState()

    def analyze_dependencies(self, plan: List[Dict]):
        """Enhanced dependency analysis with entity lifecycle tracking"""
        self.execution_state.dependency_graph.clear()
        
        # Build dependency graph
        for step in plan:
            step_id = step.get('step_id', str(uuid.uuid4()))
            self.execution_state.dependency_graph.add_node(step_id, **step)
            
            # Track entity production
            for entity in step.get('output_entities', []):
                self.execution_state.track_entity_activity(
                    entity, 'production', step)
                self.execution_state.dependency_graph.add_edge(entity, step_id)
                
            # Track entity consumption
            for param in self._get_entity_references(step):
                self.execution_state.track_entity_activity(
                    param, 'consumption', step)
                self.execution_state.dependency_graph.add_edge(step_id, param)

    def _get_entity_references(self, step: Dict) -> List[str]:
        """Extract entity references from parameters"""
        refs = []
        for param, value in step.get('parameters', {}).items():
            if isinstance(value, str) and value.startswith("$"):
                refs.append(value[1:])
        return refs