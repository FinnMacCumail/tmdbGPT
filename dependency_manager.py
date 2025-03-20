# dependency_manager.py
from typing import Dict, List, Set
import re
from nlp_retriever import execute_api_call
from typing import Optional
from utils.metadata_parser import MetadataParser

class DependencyManager:
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.resolved_entities: Dict[str, str] = {}
        self.api_results: Dict[str, Dict] = {}

    def analyze_dependencies(self, plan: List[Dict]) -> List[Dict]:
        """Build dependency graph and validate execution order"""
        step_deps = {}
        
        # First pass - identify dependencies
        for step in plan:
            step_id = f"{step['endpoint']}-{step['method']}"
            dependencies = set()
            
            # Path parameter dependencies
            path_params = re.findall(r"{(\w+_id)}", step["endpoint"])
            dependencies.update(path_params)
            
            # Query parameter dependencies
            for param in step.get("parameters", {}):
                if param.endswith("_id") and param not in self.resolved_entities:
                    dependencies.add(param)
            
            step_deps[step_id] = dependencies
        
        # Topological sort for execution order
        ordered_steps = []
        visited = set()
        
        def visit(step_id):
            if step_id not in visited:
                visited.add(step_id)
                for dep in step_deps.get(step_id, set()):
                    visit(dep)
                ordered_steps.append(next(
                    s for s in plan 
                    if f"{s['endpoint']}-{s['method']}" == step_id
                ))
        
        for step in plan:
            step_id = f"{step['endpoint']}-{step['method']}"
            visit(step_id)
            
        return ordered_steps

    def register_result(self, endpoint: str, result: Dict):
        """Extract and store resolvable entities from API responses"""
        self.api_results[endpoint] = result
        
        # Auto-extract common ID patterns
        if "results" in result:
            for item in result["results"]:
                if "id" in item:
                    entity_type = endpoint.split("/")[1]
                    self.resolved_entities[f"{entity_type}_id"] = item["id"]
                    
        if "id" in result:
            entity_type = endpoint.split("/")[1]
            self.resolved_entities[f"{entity_type}_id"] = result["id"]

    def resolve_parameters(self, parameters: Dict) -> Dict:
        """Replace entity references with resolved values"""
        resolved = {}
        for param, value in parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                entity_ref = value[1:]
                resolved[param] = self.resolved_entities.get(entity_ref, value)
            else:
                resolved[param] = value
        return resolved

# dependency_manager.py
class ExecutionState:
    def __init__(self, resolved_entities: Dict):
        self.dependency_manager = DependencyManager()
        self.dependency_manager.resolved_entities = resolved_entities  # Add this
        self.pending_steps: List[Dict] = []
        self.completed_steps: List[Dict] = []

    def add_step(self, step: Dict):
        requires = MetadataParser.parse_list(step["metadata"]["resolution_dependencies"])
        if all(req in self.entities for req in requires):
            self.pending_steps.append(step)
        else:
            print(f"⏳ Deferring {step['path']} - Missing: {requires}")

    def execute_next(self) -> Optional[Dict]:
        """Execute next available step"""
        if not self.pending_steps:
            return None
            
        step = self.pending_steps.pop(0)
        response = self._execute_step(step)
        self.completed_steps.append(step)
        return response

    def _execute_step(self, step: Dict) -> Dict:
        """Internal execution logic"""
        resolved_params = self.dependency_manager.resolve_parameters(
            step.get("parameters", {})
        )
        
        # Make API call using existing function
        response = execute_api_call({
            "endpoint": step["endpoint"],
            "method": step["method"],
            "parameters": resolved_params
        })
        
        # Update resolved entities
        self.dependency_manager.register_result(step["endpoint"], response)
        return response