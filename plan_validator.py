# plan_validator.py
from typing import List, Dict, Any
import re
from dependency_manager import ExecutionState

class PlanValidator:
    def __init__(self, resolved_entities: Dict):
        self.resolved_entities = resolved_entities

    def validate_plan(self, raw_plan: Any) -> List[Dict]:
        valid_steps = []
        
        for idx, step in enumerate(raw_plan):
            if not isinstance(step, dict):
                print(f"⚠️ Invalid step type at position {idx}: {type(step)}")
                continue
                
            # Check minimum required fields
            if not all(key in step for key in ["step_id", "endpoint"]):
                print(f"🚨 Malformed step at position {idx}: Missing step_id or endpoint")
                continue

            # Clean endpoint formatting
            endpoint = step["endpoint"]
            endpoint = endpoint.replace("{{", "{").replace("}}", "}")  # Fix JSON escapes
            endpoint = endpoint.replace("{{{", "{").replace("}}}", "}")  # Handle triple escapes
            
            # Default to GET if method not specified
            method = step.get("method", "GET")
            
            # Validate parameters format
            parameters = {}
            if "parameters" in step:
                if isinstance(step["parameters"], dict):
                    parameters = step["parameters"]
                else:
                    print(f"⚠️ Invalid parameters type in step {step['step_id']}")

            valid_steps.append({
                "step_id": step["step_id"],
                "endpoint": endpoint,
                "method": method,
                "parameters": parameters
            })
        
        return valid_steps