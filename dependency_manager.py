# dependency_manager.py
from typing import Dict, List, Set, TypedDict, Any, Optional
import re
from pydantic import BaseModel, ConfigDict
from networkx import DiGraph
import uuid
import time
import networkx as nx
from networkx.readwrite import json_graph

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
    resolved_entities: Dict[str, Any] = {}
    entity_dependencies: Dict[str, List[str]] = {}

    # Execution Tracking
    detected_intents: Dict[str, Any] = {}
    data_registry: dict[str, Any] = {}
    pending_steps: list[dict] = []
    completed_steps: list[dict] = []

    # System Internals
    entity_lifecycle: dict[str, list[EntityLifecycleEntry]] = {}
    dependency_graph: DiGraph = DiGraph()

    query_type: str = "general_info"
    specialized_params: dict = {}
    response_format: str = "standard_biography"

    # Step management
    pending_steps: List[Dict] = []
    completed_steps: List[Dict] = []

    # Data storage
    api_results: Dict[str, Any] = {}

    def track_entity_activity(self, entity: str, activity_type: str, step: Dict):
        """Record entity production or consumption"""
        entry = EntityLifecycleEntry(
            type=activity_type,
            step_id=step.get('step_id', 'unknown'),
            timestamp=time.time()
        )
        self.entity_lifecycle.setdefault(entity, []).append(entry)    
    

class DependencyManager:
    def __init__(self):
        self.execution_state = ExecutionState()
        self.graph = nx.DiGraph()

    def serialize(self) -> dict:
        """Serialize dependency graph to JSON-serializable format"""
        return json_graph.node_link_data(self.graph)

    def analyze_dependencies(self, plan: List[Dict]) -> None:
        """Build comprehensive dependency graph considering both entity and step dependencies.
        
        Args:
            plan: List of steps with potential dependencies. Each step should contain:
                - parameters: For entity dependencies ($entity syntax)
                - requires_steps: List of step_ids this step depends on (optional)
                - produces_entities: List of entities this step generates (optional)
        """
        graph = self.execution_state.dependency_graph
        graph.clear()

        # First pass: Create nodes and entity relationships
        for step in plan:
            # Generate step ID if missing
            step_id = step.setdefault('step_id', str(uuid.uuid4()))
            
            # Add node with step metadata
            graph.add_node(step_id, **step)
            
            # Track entity production
            for entity in step.get('produces_entities', []):
                graph.add_node(entity, type='entity')
                graph.add_edge(step_id, entity)

            # Entity dependencies from parameters
            required_entities = [
                p[1:] for p in step.get('parameters', {}).values()
                if isinstance(p, str) and p.startswith("$")
            ]
            for entity in required_entities:
                if entity not in graph:
                    graph.add_node(entity, type='entity')
                graph.add_edge(entity, step_id)

        # Second pass: Handle step-to-step dependencies
        for step in plan:
            step_id = step['step_id']
            
            # Explicit step dependencies
            for dep_id in step.get('requires_steps', []):
                if dep_id in graph:
                    graph.add_edge(dep_id, step_id)
                else:
                    print(f"⚠️ Missing step dependency: {dep_id} -> {step_id}")

            # Implicit dependencies via produced entities
            for entity in step.get('produces_entities', []):
                for consumer in graph.successors(entity):
                    if consumer != step_id:  # Avoid self-reference
                        graph.add_edge(step_id, consumer)
    
    def build_dependency_graph(self, steps: list):
        """Create execution order based on entity dependencies"""
        for step in steps:
            self.graph.add_node(step['step_id'])
            for dep in step.get('dependencies', []):
                if dep in self.graph:
                    self.graph.add_edge(dep, step['step_id'])