from enum import Enum, auto
from typing import Dict, List, Optional, Set
import re
import json
import uuid
from tmdbi_types import ExecutionState
import chromadb
import traceback

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

class StepType(Enum):
    ENTITY_DISCOVERY = auto()    # Creates new entity IDs (e.g., /search/person)
    DATA_RETRIEVAL = auto()      # Uses existing IDs to get data (e.g., /person/{id})
    RELATIONSHIP_MAPPING = auto()# Connects entities (e.g., /movie/{id}/credits)
    CONTENT_FILTERING = auto()   # Filters existing data (e.g., /discover/movie)

def classify_step(step: Dict) -> StepType:
    """Enhanced step classification with path pattern analysis"""
    endpoint = step.get('endpoint', '')
    
    # 1. Entity discovery endpoints
    if '/search/' in endpoint:
        return StepType.ENTITY_DISCOVERY
        
    # 2. Data retrieval endpoints with path parameters
    if any(p in step.get('parameters', {}) 
           for p in re.findall(r'{(\w+)}', endpoint)):
        return StepType.DATA_RETRIEVAL
        
    # 3. Relationship mapping endpoints
    if any(k in endpoint for k in ['/credits', '/similar', '/recommendations']):
        return StepType.RELATIONSHIP_MAPPING
        
    # 4. Fallback to content filtering
    return StepType.CONTENT_FILTERING

def _should_skip_step(raw_step: Dict, state: ExecutionState) -> bool:
    """State-aware step validation with critical exemptions"""
    step_type = classify_step(raw_step)
    
    # Never skip these critical step types
    if step_type in [StepType.DATA_RETRIEVAL, StepType.RELATIONSHIP_MAPPING]:
        return False
        
    # Original entity-based skipping logic
    step_outputs = set(raw_step.get('output_entities', []))
    return step_outputs.issubset(state.get('resolved_entities', {}).keys())

def _validate_single_step(raw_step: Dict, state: ExecutionState) -> Optional[Dict]:
    """Validate and normalize a single plan step"""
    try:
        # Validate basic structure
        required_keys = {'description', 'endpoint', 'parameters'}
        if missing_keys := required_keys - raw_step.keys():
            print(f"Missing keys: {missing_keys}")
            return None
        
        # Critical step classification
        step_type = classify_step(raw_step)
        is_critical = step_type in [StepType.DATA_RETRIEVAL, StepType.RELATIONSHIP_MAPPING]

        # Endpoint validation
        endpoint_match = _find_endpoint_match(raw_step['endpoint'])
        if not endpoint_match:
            return None if not is_critical else {  # Force critical steps
                **raw_step, 
                'metadata': {'force_execution': True, 'validated_endpoint': raw_step['endpoint']}
            }

        # Parameter validation
        validated_params = {}
        endpoint_params = json.loads(endpoint_match.get('parameters', '[]'))
        for param in endpoint_params:
            pname = param.get('name')
            if pname not in raw_step['parameters'] and param.get('required'):
                if is_critical:  # Allow default values for critical steps
                    validated_params[pname] = param.get('schema', {}).get('default')
                    print(f"⚠️ Using default for missing param in critical step: {pname}")
                else:
                    print(f"Missing required param: {pname}")
                    return None
            else:
                validated_params[pname] = raw_step['parameters'].get(pname)

        # Dependency validation
        missing_deps = [
            dep for dep in raw_step.get("depends_on", [])
            if dep not in state.get("step_status", {})
        ]
        if missing_deps and not is_critical:
            print(f"Missing dependencies: {missing_deps}")
            state.setdefault("validation_errors", []).extend([
                f"{raw_step.get('step_id')} requires {dep}" for dep in missing_deps
            ])
            return None

        # Build validated step with critical metadata
        return {
            'step_id': raw_step.get('step_id', uuid.uuid4().hex),
            'description': raw_step['description'],
            'endpoint': endpoint_match['path'],
            'method': endpoint_match.get('method', 'GET'),
            'parameters': validated_params,
            'output_entities': raw_step.get('output_entities', []),
            'dependencies': raw_step.get('depends_on', []),
            'metadata': {
                'critical': is_critical,
                'force_execution': is_critical,
                'required_entities': list(re.findall(r'\$(\w+)', str(validated_params)))
            }
        }

    except Exception as e:
        print(f"Validation error: {str(e)}")
        return None

def validate_plan_completeness(plan: List[Dict], state: ExecutionState) -> bool:
    """Ensure plan contains required data retrieval steps"""
    required_patterns = {
        'person': '/person/{person_id}',
        'movie': '/movie/{movie_id}',
        'tv': '/tv/{tv_id}'
    }
    
    present_types = set()
    for step in plan:
        endpoint = step.get('endpoint', '')
        for ent_type, pattern in required_patterns.items():
            if pattern in endpoint:
                present_types.add(ent_type)
    
    resolved_types = {
        t for t in ['person', 'movie', 'tv']
        if f"{t}_id" in state['resolved_entities']
    }
    
    return resolved_types.issubset(present_types)

def generate_validation_report(plan: List[Dict], state: ExecutionState) -> Dict:
    """Generate detailed validation feedback"""
    report = {
        'missing_data_steps': [],
        'unused_entities': [],
        'dependency_errors': state.get('validation_errors', [])
    }
    
    # Check for core detail endpoints
    for ent_type in ['person', 'movie', 'tv']:
        pattern = f"/{ent_type}/{{{ent_type}_id}}"
        if not any(pattern in step.get('endpoint', '') for step in plan):
            if f"{ent_type}_id" in state['resolved_entities']:
                report['missing_data_steps'].append(pattern)
    
    # Track unused entities
    used_entities = set()
    for step in plan:
        used_entities.update(re.findall(r'\$(\w+)', str(step['parameters'])))
    
    for entity in state['resolved_entities']:
        if entity not in used_entities:
            report['unused_entities'].append(entity)
    
    return report

def _find_endpoint_match(endpoint_pattern: str) -> Optional[Dict]:
    """
    Find the best matching API endpoint from ChromaDB embeddings
    Returns metadata for the closest matching endpoint
    """
    try:
        # Query ChromaDB with the endpoint pattern
        results = collection.query(
            query_texts=[endpoint_pattern],
            n_results=1,
            include=["metadatas", "documents"]
        )
        
        # Check for valid results
        if not results or not results['metadatas']:
            print(f"⚠️ No matches found for pattern: {endpoint_pattern}")
            return None
            
        # Extract and format the best match
        best_match = {
            "path": results['metadatas'][0][0].get("path", ""),
            "method": results['metadatas'][0][0].get("method", "GET"),
            "parameters": json.loads(
                results['metadatas'][0][0].get("parameters", "[]")
            ),
            "operation_type": results['metadatas'][0][0].get("operation_type", ""),
            "embedding_text": results['documents'][0][0]
        }
        
        # Debug matching score
        distance = results['distances'][0][0]
        print(f"🔍 Matched '{endpoint_pattern}' to '{best_match['path']}' "
              f"(confidence: {1 - distance:.2f})")
        
        return best_match
        
    except Exception as e:
        print(f"🚨 Endpoint matching failed: {str(e)}")
        traceback.print_exc()
        return None