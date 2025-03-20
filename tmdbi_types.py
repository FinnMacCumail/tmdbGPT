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

    # Entity Lifecycle Tracking
    entity_origins: Dict[str, str]  # {"person_id": "resolve_entities_step"}
    entity_dependencies: Dict[str, List[str]]  # {"movie_id": ["person_id"]}
    entity_timestamps: Dict[str, float]  # {"person_id": 1717982469.123}

    # Add entity state tracking
    entity_usage: Dict[str, List[str]]  # Track which steps USE each entity
    entity_provenance: Dict[str, Dict]  # Source of truth for entity values
    step_output_types: Dict[str, List[str]]  # Track data vs entity outputs
    
    # Execution Metadata
    step_status: Dict[str, Dict]  # {"execute_steps": {"status": "success", "duration": 2.1}}
    validation_history: List[Dict]  # Track validation decisions
    fallback_usage: Dict[str, int]  # Track fallback triggers

    # Add planning validation state
    validation_overrides: Dict[str, str]  # Manual step inclusion reasons
    llm_plan_raw: Dict  # Store original LLM plan before validation