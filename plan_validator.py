from typing import List, Dict, Any
import re
from dependency_manager import ExecutionState
from param_utils import normalize_parameters

class PlanValidator:
    def __init__(self, resolved_entities: Dict):
        self.resolved_entities = resolved_entities

    def validate_plan(self, raw_plan: Any) -> List[Dict]:
        valid_steps = []

        for idx, step in enumerate(raw_plan):
            if not isinstance(step, dict):
                print(f"⚠️ Invalid step type at position {idx}: {type(step)}")
                continue

            # 🚨 Ensure minimum fields are present
            if not all(key in step for key in ["step_id", "endpoint"]):
                print(f"🚨 Malformed step at position {idx}: Missing step_id or endpoint")
                continue

            # 🧹 Clean endpoint formatting
            endpoint = step["endpoint"]
            endpoint = endpoint.replace("{{{", "{").replace("}}}", "}")  # triple escape fix
            endpoint = endpoint.replace("{{", "{").replace("}}", "}")   # double escape fix

            # 📌 Default to GET if method missing
            method = step.get("method", "GET")

            # 🛡 Normalize parameters and sanitize early
            raw_params = step.get("parameters", {})
            parameters = normalize_parameters(raw_params)

            # 🔁 Inject resolved entity substitutions for path-style parameters
            for entity_key, entity_value in self.resolved_entities.items():
                if f"{{{entity_key}}}" in endpoint:
                    parameters[entity_key] = entity_value[0] if isinstance(entity_value, list) else entity_value

            # 🧩 Inject join-style with_* parameters where appropriate
            JOIN_PARAM_MAP = {
                "person_id": "with_people",
                "genre_id": "with_genres",
                "company_id": "with_companies",
                "keyword_id": "with_keywords",
                "network_id": "with_networks",
                "collection_id": "with_collections",
                "tv_id": "with_tv",
                "movie_id": "with_movies"
            }
            for entity_key, param_name in JOIN_PARAM_MAP.items():
                if entity_key in self.resolved_entities:
                    ids = self.resolved_entities[entity_key]
                    if isinstance(ids, list):
                        parameters[param_name] = ",".join(map(str, ids))
                    else:
                        parameters[param_name] = str(ids)

            # ✅ Append fully normalized and enriched step
            valid_steps.append({
                "step_id": step["step_id"],
                "endpoint": endpoint,
                "method": method,
                "parameters": parameters
            })

        return valid_steps
