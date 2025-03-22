# execution_orchestrator.py
from typing import Dict, List, Any
import networkx as nx
import requests
from networkx import DiGraph
from dependency_manager import ExecutionState
from utils.metadata_parser import MetadataParser

class ExecutionOrchestrator:
    def __init__(self, base_url: str, headers: Dict):
        self.base_url = base_url
        self.headers = headers
        self.retry_strategy = {
            'max_retries': 2,
            'status_codes': [429, 500, 502, 503, 504]
        }

    def execute_plan(self, state: ExecutionState) -> ExecutionState:
        """Orchestrate the execution of all steps in the plan"""
        if not state.pending_steps:
            return self._handle_empty_plan(state)

        try:
            execution_order = self._get_execution_order(state)
            for step_id in execution_order:
                state = self._execute_step(state, step_id)
        except nx.NetworkXUnfeasible:
            state.error = "Circular dependency detected in execution plan"
        
        return state

    def _get_execution_order(self, state: ExecutionState) -> List[str]:
        """Get safe execution order with cycle handling"""
        try:
            return list(nx.topological_sort(state.dependency_graph))
        except nx.NetworkXUnfeasible:
            return list(state.dependency_graph.nodes)

    def _execute_step(self, state: ExecutionState, step_id: str) -> ExecutionState:
        """Execute a single step with retry logic"""
        step_data = state.dependency_graph.nodes[step_id]
        retry_count = 0
        
        while retry_count <= self.retry_strategy['max_retries']:
            try:
                response = self._perform_api_call(step_data, state)
                state = self._process_response(state, step_id, response)
                break
            except Exception as e:
                if self._should_retry(response, retry_count):
                    retry_count += 1
                    state = self._adjust_parameters(state, step_id)
                else:
                    state.error = f"Step {step_id} failed: {str(e)}"
                    break
        return state

    def _perform_api_call(self, step_data: Dict, state: ExecutionState):
        """Execute the API call with resolved parameters"""
        resolved_params = self._resolve_parameters(step_data, state)
        validated_params = self._validate_parameter_types(step_data, resolved_params)
        
        response = requests.request(
            method=step_data.get('method', 'GET'),
            url=f"{self.base_url}{step_data['endpoint']}",
            headers=self.headers,
            params=validated_params
        )
        response.raise_for_status()
        return response

    def _resolve_parameters(self, step_data: Dict, state: ExecutionState) -> Dict:
        """Resolve parameters with entity references"""
        resolved = {}
        for param, value in step_data.get('parameters', {}).items():
            if isinstance(value, str) and value.startswith("$"):
                entity_key = value[1:]
                resolved[param] = state.resolved_entities.get(entity_key, value)
            else:
                resolved[param] = value
        return resolved

    def _validate_parameter_types(self, step_data: Dict, params: Dict) -> Dict:
        """Ensure parameters match endpoint schema types"""
        validated = {}
        for param, value in params.items():
            param_type = self._get_param_type(step_data['endpoint'], param)
            validated[param] = self._convert_type(value, param_type)
        return validated

    def _get_param_type(self, endpoint: str, param: str) -> str:
        """Get parameter type from stored metadata"""
        # Implementation would query ChromaDB collection
        return "string"  # Simplified for example

    def _convert_type(self, value: Any, target_type: str) -> Any:
        """Type conversion with error handling"""
        try:
            type_map = {
                "integer": int,
                "number": float,
                "boolean": bool
            }
            return type_map.get(target_type, str)(value)
        except ValueError:
            return value

    def _process_response(self, state: ExecutionState, step_id: str, response) -> ExecutionState:
        """Process successful API response"""
        state.data_registry[step_id] = {
            'status': response.status_code,
            'data': response.json(),
            'error': None
        }
        state.completed_steps.append(step_id)
        state.pending_steps.remove(next(
            s for s in state.pending_steps if s.get('step_id') == step_id
        ))
        return state

    def _should_retry(self, response, retry_count: int) -> bool:
        """Determine if a request should be retried"""
        return (
            retry_count < self.retry_strategy['max_retries'] and
            response.status_code in self.retry_strategy['status_codes']
        )

    def _adjust_parameters(self, state: ExecutionState, step_id: str) -> ExecutionState:
        """Adjust parameters for retry attempts"""
        # Implementation could modify query parameters or use alternate endpoints
        return state

    def _handle_empty_plan(self, state: ExecutionState) -> ExecutionState:
        """Safer error handling"""
        if not state.error:  # Only set if no existing error
            state.error = "No executable steps found - using direct entity access"
        state.data_registry['fallback'] = {
            'entities': state.resolved_entities
        }
        return state