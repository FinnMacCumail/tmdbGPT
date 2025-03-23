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
    # Add explicit type declaration for resolved_entities
    resolved_entities: Dict[str, int] = {}  # ID values should be integers

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
        """Build dependency graph for execution order"""
        self.execution_state.dependency_graph.clear()
        
        # Create node for each step
        for step in plan:
            step_id = step.get('step_id', str(uuid.uuid4()))
            self.execution_state.dependency_graph.add_node(step_id, **step)
            
            # Link dependencies
            required_entities = [
                p[1:] for p in step.get('parameters', {}).values()
                if isinstance(p, str) and p.startswith("$")
            ]
            for entity in required_entities:
                self.execution_state.dependency_graph.add_edge(entity, step_id)

    def _get_entity_references(self, step: Dict) -> List[str]:
        """Extract entity references from parameters"""
        refs = []
        for param, value in step.get('parameters', {}).items():
            if isinstance(value, str) and value.startswith("$"):
                refs.append(value[1:])
        return refs