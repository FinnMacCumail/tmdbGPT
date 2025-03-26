# execution_orchestrator.py
import requests
from typing import Dict, Any, List
from dependency_manager import ExecutionState, DependencyManager
import json
from requests.exceptions import HTTPError
import networkx as nx
from collections import defaultdict
import time
import re

class ExecutionOrchestrator:
    def __init__(self, base_url: str, headers: Dict):
        self.base_url = base_url
        self.headers = headers
        self.dependency_manager = DependencyManager()  # Add this line

    
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
        params = {}
        for param, value in step.get("parameters", {}).items():
            # Handle both $var and $$var formats
            if isinstance(value, str):
                value = value.replace("$$", "$")  # Normalize double $ signs
                
            if isinstance(value, str) and value.startswith("$"):
                entity_key = value[1:]
                params[param] = state.resolved_entities.get(entity_key)
            else:
                params[param] = value
        return params

    def _format_endpoint(self, endpoint: str, params: Dict) -> str:
        """Universal path parameter substitution"""
        # Match both {param} and $param formats
        for param_name in re.findall(r"{(\w+)}|\$(\w+)", endpoint):
            clean_param = param_name[0] or param_name[1]
            if clean_param in params:
                endpoint = endpoint.replace(
                    f"{{{clean_param}}}" if "{" in endpoint else f"${clean_param}",
                    str(params[clean_param])
                )
        return endpoint
    