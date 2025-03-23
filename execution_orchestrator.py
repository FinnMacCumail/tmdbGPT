# execution_orchestrator.py
import requests
from typing import Dict, Any, List
from dependency_manager import ExecutionState, DependencyManager
import json
from requests.exceptions import HTTPError
import networkx as nx
from collections import defaultdict
import time

class ExecutionOrchestrator:
    def __init__(self, base_url: str, headers: Dict):
        self.base_url = base_url
        self.headers = headers
        self.dependency_manager = DependencyManager()  # Add this line

    ENDPOINT_MAP = {
        'person': {
            'detail': '/person/{person_id}',
            'credits': '/person/{person_id}/movie_credits'
        },
        'movie': {
            'detail': '/movie/{movie_id}',
            'similar': '/movie/{movie_id}/similar',
            'recommendations': '/movie/{movie_id}/recommendations'
        },
        'trending': '/trending/{media_type}/{time_window}',
        'discover': '/discover/movie'
    }

    def generate_steps(self, entities: Dict, intents: Dict) -> List[Dict]:
        """Dynamically generate execution steps based on available entities"""
        steps = []
        
        # Entity-based steps
        for entity_type in ['person', 'movie', 'tv']:
            if f"{entity_type}_id" in entities:
                steps.append(self._create_detail_step(entity_type, entities))
                
        # Intent-based steps
        if 'trending' in intents.get('primary_intent', ''):
            steps.append(self._create_trending_step(entities))
        
        if 'filtered_search' in intents.get('secondary_intents', []):
            steps.append(self._create_discover_step(entities))
            
        return steps

    def _create_detail_step(self, entity_type: str, entities: Dict) -> Dict:
        return {
            "step_id": f"{entity_type}_details",
            "endpoint": self.ENDPOINT_MAP[entity_type]['detail'],
            "method": "GET",
            "parameters": {f"{entity_type}_id": f"${entity_type}_id"}
        }

    def _create_trending_step(self, entities: Dict) -> Dict:
        media_type = self._detect_media_type(entities)
        return {
            "step_id": "trending",
            "endpoint": self.ENDPOINT_MAP['trending'],
            "method": "GET",
            "parameters": {
                "media_type": media_type,
                "time_window": "week"
            }
        }

    def _create_discover_step(self, entities: Dict) -> Dict:
        params = {}
        if 'genre' in entities:
            params['with_genres'] = ",".join(entities['genre_ids'])
        if 'year' in entities:
            params['primary_release_year'] = entities['year'][0]
        return {
            "step_id": "discover",
            "endpoint": self.ENDPOINT_MAP['discover'],
            "method": "GET",
            "parameters": params
        }

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

class WorkflowOrchestrator:
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.entity_lifecycle = defaultdict(list)

    def build_plan(self, steps: List[Dict]):
        """Construct dependency graph with entity tracking"""
        self.dependency_graph.clear()
        
        for step in steps:
            step_id = step['step_id']
            self.dependency_graph.add_node(step_id, **step)
            
            # Track entity production/consumption
            if step['operation_type'] == 'entity_production':
                for entity in step['output_entities']:
                    self.entity_lifecycle[entity].append({
                        'producer': step_id,
                        'timestamp': time.time()
                    })
            
            # Create edges for dependencies
            for dep in step.get('requires_entities', []):
                if dep in self.entity_lifecycle:
                    latest_producer = self.entity_lifecycle[dep][-1]['producer']
                    self.dependency_graph.add_edge(latest_producer, step_id)