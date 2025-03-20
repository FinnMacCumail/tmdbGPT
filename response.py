from tmdbi_types import ExecutionState
from execution import direct_api_fetch

def build_response(state: ExecutionState) -> str:
    # Priority 1: Normal execution results
    if state['execution_results']:
        return format_results(state['execution_results'])
    
    # Priority 2: Direct entity fallback
    if state['resolved_entities']:
        state['direct_data_fetch'] = direct_api_fetch(
            entity_type='person',
            entity_id=state['resolved_entities'].get('person_id')
        )
        return format_fallback(state['direct_data_fetch'])
    
    # Priority 3: Error state
    return format_errors(state['validation_errors'])

def direct_api_fetch(entity_type: str, entity_id: int) -> Dict:
    """Fallback API call using resolved entity ID"""
    return execute_api_call({
        "endpoint": f"/{entity_type}/{entity_id}",
        "method": "GET",
        "parameters": {}
    })