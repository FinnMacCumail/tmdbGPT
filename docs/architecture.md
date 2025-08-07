# Architecture Documentation

This document provides an in-depth technical analysis of TMDBGPT's architecture, design patterns, and implementation strategies.

## High-Level Architecture Overview

TMDBGPT is structured as a multi-step query planner that combines **semantic search** with **symbolic constraints** to query the TMDB API. It breaks down user requests into a sequence of API calls and validations, orchestrated by a central execution loop.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  NLP Processing  │───▶│  Semantic       │
│  "Movies by X"  │    │  Entity/Intent   │    │  Endpoint       │
│                 │    │  Extraction      │    │  Retrieval      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Formatted     │◀───│   Validation &   │◀───│  Constraint     │
│   Response      │    │   Filtering      │    │  Planning &     │
│                 │    │                  │    │  Execution      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **LLM-Based Query Understanding**: Uses OpenAI models and SpaCy to parse natural language
2. **Semantic Endpoint Retrieval**: ChromaDB vector store for finding relevant API endpoints
3. **Symbolic Constraint Planning**: Constraint trees for precise filtering and validation
4. **Dynamic Multi-Role Support**: Handles complex queries with multiple people and roles
5. **Robust Execution Loop**: Centralized orchestration with fallback mechanisms
6. **Fallback & Validation**: Progressive constraint relaxation and result validation

## Core Architecture Layers

### Layer 1: Natural Language Processing

**Location**: `nlp/nlp_retriever.py`, `core/llm/extractor.py`

**Purpose**: Translates natural language queries into structured data

**Components**:
- **Entity Extraction**: Identifies people, movies, genres, companies
- **Intent Classification**: Determines query type (summary, count, fact, list)
- **Role Detection**: Recognizes roles (director, actor, writer, etc.)

**Example**:
```
Input: "Movies directed by Christopher Nolan and starring Leonardo DiCaprio"

Output: {
  "query_entities": [
    {"name": "Christopher Nolan", "type": "person", "role": "director"},
    {"name": "Leonardo DiCaprio", "type": "person", "role": "actor"}
  ],
  "intent": "summary.movie",
  "question_type": "summary",
  "media_type": "movie"
}
```

### Layer 2: Semantic Search Engine

**Location**: `core/embeddings/hybrid_retrieval.py`, `core/embeddings/semantic_embed.py`

**Purpose**: Finds relevant TMDB API endpoints using vector similarity

**Components**:
- **ChromaDB Vector Store**: Stores embedded endpoint descriptions
- **Sentence Transformers**: Converts queries to vector embeddings
- **Semantic Matching**: Finds best endpoint candidates

**Process**:
1. Query embedded using `all-MiniLM-L6-v2` model
2. Vector similarity search in ChromaDB
3. Top-k endpoint candidates retrieved
4. Ranked by semantic relevance score

### Layer 3: Constraint Planning System

**Location**: `core/model/constraint.py`, `core/planner/`

**Purpose**: Builds symbolic representation of query requirements

**Components**:
- **Constraint Tree**: Hierarchical representation of filters
- **Entity Resolution**: Maps names to TMDB IDs
- **Plan Generation**: Creates multi-step execution plan

**Constraint Tree Example**:
```python
ConstraintGroup([
    Constraint("with_people", [6193, 1233]),  # DiCaprio, Nolan IDs
    Constraint("with_genres", [878]),         # Sci-fi genre
    Constraint("primary_release_year", 2010)
], logic="AND")
```

### Layer 4: Execution Engine

**Location**: `core/execution/step_runner.py`, `core/execution/execution_orchestrator.py`

**Purpose**: Executes planned API calls and handles results

**Components**:
- **StepRunner**: Iterates through execution steps
- **Discovery Handler**: Processes broad search results
- **Result Handler**: Processes specific item lookups
- **Fallback Manager**: Handles failed queries

**Execution Flow**:
1. Step validation and parameter injection
2. TMDB API request execution
3. Result processing and enrichment
4. Constraint validation
5. Result accumulation

### Layer 5: Validation & Filtering

**Location**: `core/validation/role_validators.py`, `core/execution/post_validator.py`

**Purpose**: Ensures results meet all query requirements

**Components**:
- **Role Validators**: Verify person roles in credits
- **Symbolic Filters**: Apply constraint tree filtering
- **Post-Validation**: Double-check results against requirements

## Data Flow Architecture

### 1. Query Ingestion

```
User Query → Entity Extraction → Intent Classification → Constraint Building
```

### 2. Planning Phase

```
Constraints → Semantic Search → Endpoint Ranking → Plan Generation
```

### 3. Execution Phase

```
Plan Steps → API Calls → Result Processing → Validation → Accumulation
```

### 4. Fallback Handling

```
Empty Results → Constraint Relaxation → Semantic Fallback → Final Results
```

## Key Design Patterns

### 1. Multi-Stage Pipeline

Each query progresses through distinct stages:
- **Parse**: Basic query understanding
- **Extract**: Entity and intent extraction  
- **Resolve**: Entity ID resolution
- **Retrieve**: Semantic endpoint matching
- **Plan**: Execution plan generation
- **Execute**: API call execution
- **Respond**: Result formatting

### 2. Constraint Satisfaction

**Symbolic Constraints**: Hard requirements that must be met
```python
# All movies must have both people in specified roles
constraints = [
    ("with_people", [person1_id, person2_id]),
    ("person_role", {"person1_id": "director", "person2_id": "actor"})
]
```

**Soft Constraints**: Preferences that can be relaxed
```python
# Preferred but not required
soft_constraints = [
    ("with_genres", [genre_id]),
    ("primary_release_year", year)
]
```

### 3. Role-Aware Processing

The system understands and validates specific roles:

```python
# Director validation
def has_director(movie_credits, director_name):
    for crew_member in movie_credits.get("crew", []):
        if crew_member.get("job") == "Director":
            if director_name.lower() in crew_member.get("name", "").lower():
                return True
    return False
```

### 4. Progressive Fallback Strategy

When queries fail, the system tries:

1. **Constraint Relaxation**: Drop least important filters
2. **Parameter Adjustment**: Modify search parameters  
3. **Semantic Fallback**: Broaden search semantically
4. **Default Results**: Provide related content

```python
# Relaxation priority order
RELAXATION_ORDER = [
    "with_companies",    # First to drop
    "with_networks", 
    "with_genres",
    "primary_release_year",
    "with_people"        # Last to drop (CORE requirement)
]
```

## Component Deep Dive

### Entity Resolution System

**Location**: `core/entity/entity_resolution.py`

**Purpose**: Maps natural language entity names to TMDB IDs

**Process**:
1. **Fuzzy Matching**: Handles spelling variations
2. **Disambiguation**: Resolves name conflicts
3. **ID Caching**: Stores resolved mappings
4. **Role Assignment**: Maps people to their roles

### Semantic Embedding System

**Location**: `core/embeddings/semantic_embed.py`

**Purpose**: Creates vector representations for semantic search

**Models Used**:
- **Sentence Transformers**: `all-MiniLM-L6-v2`
- **ChromaDB**: Vector storage and retrieval
- **Custom Embeddings**: Domain-specific enhancements

### Execution Orchestrator

**Location**: `core/execution/execution_orchestrator.py`

**Purpose**: Manages the overall execution workflow

**Responsibilities**:
- Step scheduling and ordering
- Resource management
- Error handling and recovery
- Progress tracking and logging

### Response Formatting System

**Location**: `core/formatting/`

**Purpose**: Converts raw results into user-friendly format

**Components**:
- **Template System**: Flexible response templates
- **Renderer Registry**: Different output formats
- **Formatting Pipeline**: Multi-stage result processing

## Advanced Features

### Multi-Entity Intersection

For queries with multiple people:

```python
# Find movies with both people
person1_movies = get_person_credits(person1_id)
person2_movies = get_person_credits(person2_id) 
intersection = person1_movies & person2_movies
```

### Dynamic Plan Expansion

The system can inject additional steps based on discovered data:

```python
# If genre discovered, add genre-specific search
if "genre_id" in resolved_entities:
    additional_steps.append({
        "endpoint": "/discover/movie",
        "parameters": {"with_genres": resolved_entities["genre_id"]}
    })
```

### Constraint Tree Evaluation

Complex logical operations on constraints:

```python
# AND/OR logic support
constraint_tree = ConstraintGroup([
    ConstraintGroup([person1, person2], logic="OR"),   # Either person
    ConstraintGroup([genre1, genre2], logic="AND")     # Both genres
], logic="AND")
```

## Performance Considerations

### Caching Strategy

- **Entity Resolution**: Cache name-to-ID mappings
- **API Responses**: Cache frequent API calls
- **Embeddings**: Cache vector representations

### Memory Management

- **ChromaDB**: Efficient vector storage
- **Batch Processing**: Handle large result sets
- **Lazy Loading**: Load data only when needed

### Scalability

- **Async Processing**: Non-blocking API calls
- **Connection Pooling**: Efficient HTTP connections
- **Rate Limiting**: Respect API limits

## Error Handling & Resilience

### Graceful Degradation

1. **API Failures**: Fall back to cached data
2. **Parsing Errors**: Use simpler extraction methods
3. **Empty Results**: Provide related suggestions
4. **Network Issues**: Retry with backoff

### Logging & Observability

- **Execution Trace**: Complete step-by-step logging
- **Performance Metrics**: Query processing times
- **Error Tracking**: Detailed error information
- **Debug Modes**: Multiple verbosity levels

## Security Considerations

### API Key Management

- Environment variable storage
- No hard-coded credentials
- Secure transmission only

### Input Validation

- Query sanitization
- Parameter validation
- Injection prevention

### Rate Limiting

- Respect TMDB API limits
- Implement backoff strategies
- Monitor usage patterns

## Extending the Architecture

### Adding New Data Sources

1. Create new resolver in `core/entity/`
2. Add endpoint definitions
3. Update semantic embeddings
4. Implement result processors

### Adding New Query Types

1. Extend intent classification
2. Add constraint types
3. Implement validators
4. Create response formatters

### Custom Validation Rules

1. Implement in `core/validation/`
2. Register with validation pipeline
3. Add constraint support
4. Update documentation

---

This architecture enables TMDBGPT to handle complex natural language queries about movies and TV shows with high accuracy and reliability, while maintaining extensibility for future enhancements.