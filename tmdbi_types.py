from typing import TypedDict, List, Dict, Any, Optional

class ExecutionState(TypedDict):
    # Core Flow
    query: str
    raw_entities: Dict[str, List[str]]
    resolved_entities: Dict[str, Any]
    detected_intents: Dict[str, Any]
    
    # Planning & Execution
    raw_plan: List[Dict[str, Any]]
    validated_plan: List[Dict[str, Any]]
    execution_results: Dict[str, Dict[str, Any]]
    
    # Fallback State
    direct_data_fetch: Optional[Dict[str, Any]]
    fallback_triggered: bool
    validation_errors: List[str]
    
    # Temporal Tracking
    execution_start: Optional[float]
    execution_duration: Optional[float]