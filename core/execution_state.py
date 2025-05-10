from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Set
# adjust import path if needed
from core.constraint_model import Constraint, ConstraintGroup


class AppState(BaseModel):
    input: str
    status: Optional[str] = None
    step: Optional[str] = None
    query: Optional[str] = None

    # Core pipeline fields
    extraction_result: Optional[Dict] = Field(default_factory=dict)
    resolved_entities: Optional[Dict] = Field(default_factory=dict)
    retrieved_matches: Optional[List] = Field(default_factory=list)
    plan_steps: Optional[List[Dict]] = Field(default_factory=list)
    responses: Optional[List[Any]] = Field(default_factory=list)
    formatted_response: Optional[Any] = None
    error: Optional[str] = None

    # Execution tracking
    data_registry: Optional[Dict] = Field(default_factory=dict)
    completed_steps: Optional[List[str]] = Field(default_factory=list)
    pending_steps: Optional[List[Dict]] = Field(default_factory=list)
    execution_trace: Optional[List[Dict]] = Field(default_factory=list)
    visited_fingerprints: Set[str] = Field(default_factory=set)

    # Planner guidance
    question_type: Optional[str] = None
    response_format: Optional[str] = None
    intended_media_type: Optional[str] = None

    # Symbolic planning & validation
    constraint_tree: Optional[ConstraintGroup] = None
    constraint_tree_evaluated: bool = False
    last_dropped_constraints: Optional[List[Constraint]] = []
    relaxation_log: List[str] = Field(default_factory=list)
    relaxed_parameters: Optional[List[str]] = Field(default_factory=list)
    post_validation_log: Optional[List[str]] = Field(default_factory=list)
    satisfied_roles: Set[str] = Field(default_factory=set)

    # Debug / system features
    explanation: Optional[str] = None
    debug: Optional[bool] = True

    class Config:
        arbitrary_types_allowed = True
