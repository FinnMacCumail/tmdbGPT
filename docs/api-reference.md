# API Reference Documentation

This document provides comprehensive reference for TMDBGPT's internal APIs, modules, and extension points for developers.

## Core Application Interface

### Main Application (`app.py`)

#### Graph Execution Functions

```python
def parse(state: AppState) -> AppState:
    """
    Initial parsing step with user-friendly progress indication.
    
    Args:
        state: Current application state with user input
        
    Returns:
        AppState: Updated state with parsing status
    """
    
def extract_entities(state: AppState) -> AppState:
    """
    Extract entities and intents from user query.
    
    Args:
        state: Application state with input query
        
    Returns:
        AppState: State with extracted entities and intent information
    """

def resolve_entities(state: AppState) -> AppState:
    """
    Resolve entity names to TMDB IDs.
    
    Args:
        state: State with extracted entities
        
    Returns:
        AppState: State with resolved entity IDs
    """

def retrieve_context(state: AppState) -> AppState:
    """
    Perform semantic search for relevant API endpoints.
    
    Args:
        state: State with resolved entities
        
    Returns:
        AppState: State with retrieved endpoint matches
    """

def plan(state: AppState) -> AppState:
    """
    Generate execution plan with constraint-aware planning.
    
    Args:
        state: State with context and constraints
        
    Returns:
        AppState: State with planned execution steps
    """

def execute(state: AppState) -> AppState:
    """
    Execute the planned API calls.
    
    Args:
        state: State with execution plan
        
    Returns:
        AppState: State with execution results
    """

def respond(state: AppState) -> dict:
    """
    Format and return final response.
    
    Args:
        state: State with execution results
        
    Returns:
        dict: Formatted response data
    """
```

#### Graph Construction

```python
def build_app_graph() -> CompiledGraph:
    """
    Build and compile the LangGraph execution graph.
    
    Returns:
        CompiledGraph: Compiled state graph for query execution
    """
```

## Core Modules

### Entity Resolution (`core/entity/entity_resolution.py`)

#### TMDBEntityResolver

```python
class TMDBEntityResolver:
    """Resolves entity names to TMDB IDs with fuzzy matching and caching."""
    
    def __init__(self, api_key: str, headers: dict):
        """
        Initialize entity resolver.
        
        Args:
            api_key: TMDB API key
            headers: HTTP headers for requests
        """
        
    def resolve_entities(self, 
                        query_entities: List[dict], 
                        intended_media_type: str = None) -> Tuple[List[dict], List[dict]]:
        """
        Resolve multiple entities from query.
        
        Args:
            query_entities: List of entity dictionaries
            intended_media_type: Target media type ("movie", "tv", "both")
            
        Returns:
            Tuple[List[dict], List[dict]]: Resolved and unresolved entities
        """
        
    def resolve_person(self, name: str) -> Optional[dict]:
        """
        Resolve person name to TMDB person ID.
        
        Args:
            name: Person's name
            
        Returns:
            Optional[dict]: Person details with ID if found
        """
        
    def resolve_movie(self, title: str) -> Optional[dict]:
        """
        Resolve movie title to TMDB movie ID.
        
        Args:
            title: Movie title
            
        Returns:
            Optional[dict]: Movie details with ID if found
        """
```

### Execution Engine (`core/execution/step_runner.py`)

#### StepRunner

```python
class StepRunner:
    """Executes planned API steps with validation and fallback handling."""
    
    def __init__(self, base_url: str, headers: dict):
        """
        Initialize step runner.
        
        Args:
            base_url: TMDB API base URL
            headers: HTTP headers for API requests
        """
        
    def execute(self, state: AppState) -> AppState:
        """
        Execute all planned steps in the application state.
        
        Args:
            state: Application state with planned steps
            
        Returns:
            AppState: Updated state with execution results
        """
        
    def finalize(self, state: AppState) -> AppState:
        """
        Finalize execution with filtering and response formatting.
        
        Args:
            state: State after step execution
            
        Returns:
            AppState: Final state with formatted results
        """
```

### Semantic Search (`core/embeddings/hybrid_retrieval.py`)

#### Core Functions

```python
def rank_and_score_matches(extraction_result: dict) -> List[dict]:
    """
    Perform semantic search and rank endpoint matches.
    
    Args:
        extraction_result: Extracted entities and intents
        
    Returns:
        List[dict]: Ranked list of matching endpoints
    """

def convert_matches_to_execution_steps(matches: List[dict], 
                                     extraction_result: dict,
                                     resolved_entities: dict) -> List[dict]:
    """
    Convert endpoint matches to executable steps.
    
    Args:
        matches: Ranked endpoint matches
        extraction_result: Query extraction data
        resolved_entities: Resolved entity mappings
        
    Returns:
        List[dict]: Executable step definitions
    """
```

### Constraint System (`core/model/constraint.py`)

#### Constraint Classes

```python
class Constraint:
    """Individual constraint representation."""
    
    def __init__(self, key: str, value: Any, operator: str = "equals"):
        """
        Create a constraint.
        
        Args:
            key: Constraint parameter name
            value: Constraint value
            operator: Comparison operator
        """
        
    def evaluate(self, data: dict) -> bool:
        """
        Evaluate constraint against data.
        
        Args:
            data: Data to evaluate against
            
        Returns:
            bool: True if constraint is satisfied
        """

class ConstraintGroup:
    """Group of constraints with logical operations."""
    
    def __init__(self, constraints: List[Constraint], logic: str = "AND"):
        """
        Create constraint group.
        
        Args:
            constraints: List of constraints
            logic: Logical operator ("AND", "OR")
        """
        
    def evaluate(self, data: dict) -> bool:
        """
        Evaluate all constraints in group.
        
        Args:
            data: Data to evaluate against
            
        Returns:
            bool: True if group evaluation passes
        """
```

### Validation System (`core/validation/role_validators.py`)

#### Role Validation Functions

```python
def has_director(movie_credits: dict, director_name: str) -> bool:
    """
    Check if movie has specified director.
    
    Args:
        movie_credits: Movie credits from TMDB API
        director_name: Director name to check
        
    Returns:
        bool: True if director found in credits
    """

def has_writer(movie_credits: dict, writer_name: str) -> bool:
    """
    Check if movie has specified writer.
    
    Args:
        movie_credits: Movie credits from TMDB API  
        writer_name: Writer name to check
        
    Returns:
        bool: True if writer found in credits
    """

def validate_roles(credits: dict, query_entities: List[dict]) -> dict:
    """
    Validate all roles from query against credits.
    
    Args:
        credits: Credits data from TMDB API
        query_entities: Entities with roles from query
        
    Returns:
        dict: Validation results for each role
    """
```

### Fallback System (`core/execution/fallback.py`)

#### FallbackHandler

```python
class FallbackHandler:
    """Handles query fallback and constraint relaxation."""
    
    @staticmethod
    def relax_constraints(step: dict) -> Optional[dict]:
        """
        Create relaxed version of failed step.
        
        Args:
            step: Original step that failed
            
        Returns:
            Optional[dict]: Relaxed step or None if can't relax
        """
        
    @staticmethod
    def inject_credit_fallback_steps(state: AppState, step: dict):
        """
        Inject role-based fallback steps.
        
        Args:
            state: Current application state
            step: Current step being processed
        """
```

### Response Formatting (`core/formatting/formatter.py`)

#### ResponseFormatter

```python
class ResponseFormatter:
    """Formats execution results into user-friendly responses."""
    
    @staticmethod
    def format_responses(state: AppState) -> List[str]:
        """
        Format state results into response lines.
        
        Args:
            state: Application state with results
            
        Returns:
            List[str]: Formatted response lines
        """
        
    @staticmethod
    def format_movie_summary(movie: dict) -> str:
        """
        Format single movie into summary string.
        
        Args:
            movie: Movie data dictionary
            
        Returns:
            str: Formatted movie summary
        """
```

## State Management

### Application State (`core/execution_state.py`)

#### AppState

```python
class AppState(BaseModel):
    """Main application state container."""
    
    # Input and processing
    input: str = ""
    step: str = "init"
    
    # Entity extraction and resolution
    extraction_result: dict = {}
    resolved_entities: dict = {}
    intended_media_type: Optional[str] = None
    
    # Planning and constraints
    constraint_tree: Optional[ConstraintGroup] = None
    retrieved_matches: List[dict] = []
    plan_steps: List[dict] = []
    
    # Execution tracking
    completed_steps: List[str] = []
    data_registry: dict = {}
    
    # Results and responses
    responses: List[dict] = []
    formatted_response: Optional[str] = None
    
    # Error handling and fallback
    error: Optional[str] = None
    relaxed_parameters: List[str] = []
    
    def model_copy(self, **kwargs) -> 'AppState':
        """Create copy of state with updates."""
```

## Extension APIs

### Custom Entity Resolvers

```python
class CustomEntityResolver(TMDBEntityResolver):
    """Example custom entity resolver."""
    
    def resolve_custom_entity_type(self, name: str) -> Optional[dict]:
        """
        Resolve custom entity type.
        
        Args:
            name: Entity name
            
        Returns:
            Optional[dict]: Resolution result
        """
        # Custom resolution logic
        pass
```

### Custom Validators

```python
def custom_validator(result: dict, constraints: dict) -> Tuple[bool, dict]:
    """
    Custom validation function.
    
    Args:
        result: Result to validate
        constraints: Constraints to check against
        
    Returns:
        Tuple[bool, dict]: (is_valid, validation_details)
    """
    # Custom validation logic
    return True, {"validated_fields": []}
```

### Custom Formatters

```python
def custom_formatter(state: AppState) -> List[str]:
    """
    Custom response formatter.
    
    Args:
        state: Application state with results
        
    Returns:
        List[str]: Formatted response lines
    """
    # Custom formatting logic
    return ["Custom formatted response"]
```

## Utility APIs

### Logging (`response/log_summary.py`)

#### Log Summary Function

```python
def log_summary(state: AppState, header: Optional[str] = None):
    """
    Generate comprehensive debugging summary.
    
    Args:
        state: Application state to summarize
        header: Optional header text
    """
```

### Trace Logging (`core/execution/trace_logger.py`)

#### ExecutionTraceLogger

```python
class ExecutionTraceLogger:
    """Logs detailed execution traces for debugging."""
    
    @staticmethod
    def log_step(step_id: str, 
                path: str, 
                status: str, 
                summary: str = "",
                state: Optional[AppState] = None):
        """
        Log execution step details.
        
        Args:
            step_id: Unique step identifier
            path: API endpoint path
            status: Step execution status
            summary: Brief summary of step
            state: Optional application state
        """
```

## Configuration APIs

### Environment Configuration

```python
def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get configuration value from environment.
    
    Args:
        key: Configuration key
        default: Default value if not found
        
    Returns:
        Any: Configuration value
    """
    return os.getenv(key, default)

def validate_required_config():
    """Validate all required configuration is present."""
    required = ["TMDB_API_KEY", "OPENAI_API_KEY"]
    for key in required:
        if not os.getenv(key):
            raise ValueError(f"Required config {key} not found")
```

## Testing APIs

### Mock State Creation

```python
def create_mock_state(input_query: str = "test query") -> AppState:
    """
    Create mock application state for testing.
    
    Args:
        input_query: Test query string
        
    Returns:
        AppState: Mock state for testing
    """
    return AppState(
        input=input_query,
        extraction_result={"entities": [], "intents": []},
        resolved_entities={},
        plan_steps=[],
        responses=[]
    )
```

### Test Utilities

```python
def mock_tmdb_response(endpoint: str, response_data: dict):
    """Mock TMDB API response for testing."""
    
def assert_valid_state(state: AppState):
    """Assert state is valid for testing."""
    
def compare_states(state1: AppState, state2: AppState) -> List[str]:
    """Compare two states and return differences."""
```

## Error Handling APIs

### Exception Classes

```python
class TMDBGPTError(Exception):
    """Base exception for TMDBGPT errors."""
    
class EntityResolutionError(TMDBGPTError):
    """Raised when entity resolution fails."""
    
class ConstraintValidationError(TMDBGPTError):
    """Raised when constraint validation fails."""
    
class APIError(TMDBGPTError):
    """Raised when API calls fail."""
```

### Error Recovery

```python
def handle_api_error(error: APIError) -> Optional[dict]:
    """
    Handle API errors with fallback strategies.
    
    Args:
        error: API error to handle
        
    Returns:
        Optional[dict]: Fallback response or None
    """
```

## Performance APIs

### Caching

```python
def cache_key_for_entity(name: str, entity_type: str) -> str:
    """Generate cache key for entity resolution."""
    
def cache_key_for_query(extraction_result: dict) -> str:
    """Generate cache key for query results."""
```

### Metrics

```python
def measure_execution_time(func):
    """Decorator to measure function execution time."""
    
def track_api_usage(endpoint: str, response_size: int):
    """Track API usage metrics."""
```

---

This API reference provides the foundation for extending, customizing, and integrating with TMDBGPT. All APIs are designed with extensibility and maintainability in mind.