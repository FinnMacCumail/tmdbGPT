# execution_orchestrator.py
import requests
from typing import Dict, Any
from dependency_manager import ExecutionState
import json
from requests.exceptions import HTTPError

class ExecutionOrchestrator:
    def __init__(self, base_url: str, headers: Dict):
        self.base_url = base_url
        self.headers = headers
        
    def execute(self, state: ExecutionState) -> ExecutionState:
        """Execute all API steps with proper parameter handling"""
        state.error = None
        state.data_registry = {}
        
        for step in state.pending_steps:
            try:
                print(f"\n⚡ Executing {step['step_id']}: {step['endpoint']}")
                
                # Resolve parameters
                resolved_params = self._resolve_parameters(step, state)
                
                # Format endpoint URL
                formatted_endpoint = self._format_endpoint(step["endpoint"], resolved_params)
                
                # Execute API call
                response = requests.request(
                    method=step.get("method", "GET"),
                    url=f"{self.base_url}{formatted_endpoint}",
                    headers=self.headers,
                    params=resolved_params
                )
                response.raise_for_status()
                
                # Store results
                state.data_registry[step["step_id"]] = response.json()
                state.completed_steps.append(step["step_id"])
                
                print(f"✅ Success: {response.status_code}")
                
            except Exception as e:
                state.error = f"Step {step['step_id']} failed: {str(e)}"
                print(f"🔥 Error: {state.error}")
        
        return state

    def _resolve_parameters(self, step: Dict, state: ExecutionState) -> Dict:
        """Resolve $entity references to actual values"""
        params = {}
        for param, value in step.get("parameters", {}).items():
            if isinstance(value, str) and value.startswith("$"):
                entity_key = value[1:]
                if entity_key not in state.resolved_entities:
                    raise ValueError(f"Missing entity: {entity_key}")
                params[param] = state.resolved_entities[entity_key]
            else:
                params[param] = value
        return params

    def _format_endpoint(self, endpoint: str, params: Dict) -> str:
        """Replace path parameters in endpoint URL"""
        for key, value in params.items():
            endpoint = endpoint.replace(f"{{{key}}}", str(value))
        return endpoint
    
    def _execute_single_step(self, step: Dict, state: ExecutionState) -> None:
        """Execute a single API step with comprehensive error handling"""
        step_id = step.get("step_id", "unknown_step")
        
        try:
            # Resolve and validate parameters
            resolved_params = self._resolve_parameters(step, state)
            self._validate_step_parameters(step, resolved_params)
            
            # Prepare request components
            url, params = self._prepare_request(step, resolved_params)
            
            # Execute API request
            response = self._execute_request(
                method=step.get("method", "GET"),
                url=url,
                params=params
            )
            
            # Handle and store response
            response_data = self._handle_response(response)
            self._store_results(step_id, response_data, state)

        except Exception as e:
            self._handle_error(step_id, e, state)

    def _validate_step_parameters(self, step: Dict, params: Dict) -> None:
        """Validate required parameters exist in resolved values"""
        endpoint = step["endpoint"]
        required_params = [p.split("}")[0] for p in endpoint.split("{")[1:]]
        
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing path parameter: {param}")

    def _prepare_request(self, step: Dict, params: Dict) -> tuple[str, Dict]:
        """Prepare final URL and query parameters"""
        endpoint = step["endpoint"].format(**params)
        query_params = {
            k: v for k, v in params.items() 
            if k not in endpoint  # Exclude path parameters
        }
        return f"{self.base_url}{endpoint}", query_params

    def _execute_request(self, method: str, url: str, params: Dict) -> requests.Response:
        """Execute HTTP request with error handling"""
        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        return response

    def _handle_response(self, response: requests.Response) -> Dict:
        """Parse and validate response structure"""
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"raw_response": response.text}

    def _store_results(self, step_id: str, data: Dict, state: ExecutionState) -> None:
        """Store results in execution state"""
        state.data_registry[step_id] = {
            "status": "success",
            "data": data,
            "error": None
        }
        state.completed_steps.append(step_id)

    def _handle_error(self, step_id: str, error: Exception, state: ExecutionState) -> None:
        """Handle and record execution errors"""
        error_msg = f"Step {step_id} failed: {str(error)}"
        
        if isinstance(error, HTTPError):
            error_msg += f" (HTTP {error.response.status_code})"
            if error.response.content:
                error_msg += f"\nResponse: {error.response.text[:200]}..."
        elif isinstance(error, json.JSONDecodeError):
            error_msg += f" (Invalid JSON)"
        
        state.error = error_msg
        state.data_registry[step_id] = {
            "status": "error",
            "data": None,
            "error": error_msg
        }
        print(f"🔥 {error_msg}")