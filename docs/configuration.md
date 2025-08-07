# Configuration Guide

This guide covers all configuration options, environment variables, and customization settings for TMDBGPT.

## Environment Variables

### Required Configuration

Create a `.env` file in the project root directory with the following required variables:

```env
# TMDB API Configuration (Required)
TMDB_API_KEY=your_tmdb_bearer_token_here

# OpenAI API Configuration (Required)  
OPENAI_API_KEY=your_openai_api_key_here
```

### Optional Configuration

```env
# Alternative TMDB API Key format
NON_B_TMDB_API_KEY=your_tmdb_v3_api_key_here

# OpenAI Model Configuration
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.1

# ChromaDB Configuration
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION_NAME=tmdb_endpoints

# Application Settings
APP_DEBUG_MODE=false
APP_LOG_LEVEL=INFO
APP_MAX_RESULTS=20
```

## Application Configuration

### Debug Mode Settings

Control the level of debugging output by editing `app.py`:

```python
# Debug mode control - Set to True for development debugging
DEBUG_MODE = False  # User-friendly mode (default)
# DEBUG_MODE = True   # Developer debugging mode
```

**User Mode (DEBUG_MODE = False)**:
- Clean progress indicators
- Formatted results only
- No technical debugging output

**Debug Mode (DEBUG_MODE = True)**:
- 🧠 DEBUGGING SUMMARY REPORT
- Detailed execution traces
- Constraint analysis
- Step-by-step processing logs

### Query Processing Settings

Edit these constants in `app.py`:

```python
# Safe optional parameters for semantic enrichment
SAFE_OPTIONAL_PARAMS = {
    "vote_average.gte", 
    "vote_count.gte", 
    "primary_release_year",
    "release_date.gte", 
    "with_runtime.gte", 
    "with_runtime.lte",
    "with_original_language", 
    "region"
}
```

## Component Configuration

### ChromaDB Settings

**Location**: Various files in `core/embeddings/`

**Database Path**: Default `./chroma_db/`
```python
# To change database location
CHROMA_DB_PATH = "/custom/path/to/chroma_db"
```

**Collection Configuration**:
```python
# Collection for TMDB endpoints
COLLECTION_NAME = "tmdb_endpoints"
VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 dimension
```

**Memory Settings**:
```python
# Adjust for system resources
MAX_BATCH_SIZE = 100
EMBEDDING_CACHE_SIZE = 1000
```

### Sentence Transformers Configuration

**Location**: `core/embeddings/semantic_embed.py`

**Default Model**: `all-MiniLM-L6-v2`

To use a different model:
```python
# Edit in semantic_embed.py
MODEL_NAME = "all-mpnet-base-v2"  # Higher quality, slower
# MODEL_NAME = "all-distilroberta-v1"  # Good balance
# MODEL_NAME = "all-MiniLM-L6-v2"    # Faster, smaller
```

**Device Configuration**:
```python
# Force CPU usage
DEVICE = "cpu"

# Use GPU if available (automatic detection)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### LLM Configuration

**Location**: `core/llm/llm_client.py`

**OpenAI Model Settings**:
```python
MODEL = "gpt-3.5-turbo"        # Default, cost-effective
# MODEL = "gpt-4"              # Higher quality, more expensive
# MODEL = "gpt-4-turbo"        # Latest, balanced option

MAX_TOKENS = 1000              # Maximum response tokens
TEMPERATURE = 0.1              # Low temperature for consistency
```

**Request Configuration**:
```python
REQUEST_TIMEOUT = 30           # Seconds
MAX_RETRIES = 3               # Retry failed requests
BACKOFF_FACTOR = 2            # Exponential backoff
```

### SpaCy Configuration

**Location**: `nlp/nlp_retriever.py`

**Model Selection**:
```python
# Default English model
NLP_MODEL = "en_core_web_sm"   # Small, fast model

# Alternative models (require separate download)
# NLP_MODEL = "en_core_web_md"   # Medium model with vectors
# NLP_MODEL = "en_core_web_lg"   # Large model, best accuracy
```

**Processing Pipeline**:
```python
# Customize NLP pipeline components
NLP_PIPELINE = ["tok2vec", "tagger", "parser", "ner"]
DISABLE_COMPONENTS = ["lemmatizer"]  # Disable unused components
```

## Performance Tuning

### Memory Optimization

**ChromaDB Memory Usage**:
```python
# Reduce memory footprint
CHROMA_SETTINGS = {
    "anonymized_telemetry": False,
    "is_persistent": True,
    "persist_directory": "./chroma_db"
}
```

**Sentence Transformers Memory**:
```python
# Use smaller batch sizes for limited memory
EMBEDDING_BATCH_SIZE = 16      # Reduce for low memory systems
MAX_SEQ_LENGTH = 256          # Truncate long texts
```

**System Resource Limits**:
```python
# Limit concurrent operations
MAX_CONCURRENT_REQUESTS = 5
REQUEST_POOL_SIZE = 10
WORKER_THREADS = 4
```

### Speed Optimization

**Caching Configuration**:
```python
# Enable aggressive caching
CACHE_ENTITY_RESOLUTIONS = True
CACHE_API_RESPONSES = True
CACHE_EMBEDDINGS = True

# Cache sizes
ENTITY_CACHE_SIZE = 1000
API_CACHE_SIZE = 500
EMBEDDING_CACHE_SIZE = 2000
```

**API Request Optimization**:
```python
# Connection pooling
CONNECTION_POOL_SIZE = 10
KEEP_ALIVE_TIMEOUT = 30

# Request batching
BATCH_API_REQUESTS = True
MAX_BATCH_SIZE = 20
```

### Quality vs Speed Trade-offs

**High Quality Configuration**:
```python
# Best results, slower processing
SENTENCE_MODEL = "all-mpnet-base-v2"
OPENAI_MODEL = "gpt-4"
MAX_SEARCH_RESULTS = 50
ENABLE_DEEP_VALIDATION = True
```

**High Speed Configuration**:
```python
# Faster processing, good quality
SENTENCE_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-3.5-turbo"
MAX_SEARCH_RESULTS = 20
ENABLE_DEEP_VALIDATION = False
```

## Logging Configuration

### Log Levels

Edit in relevant modules:
```python
import logging

# Set global log level
logging.basicConfig(level=logging.INFO)

# Component-specific logging
logger = logging.getLogger("tmdbgpt")
logger.setLevel(logging.DEBUG)
```

**Available Levels**:
- `DEBUG`: Detailed debugging information
- `INFO`: General operational information  
- `WARNING`: Warning messages
- `ERROR`: Error conditions
- `CRITICAL`: Critical errors

### Execution Trace Logging

**Location**: `core/execution/trace_logger.py`

```python
# Enable detailed execution tracing
ENABLE_TRACE_LOGGING = True
TRACE_LOG_LEVEL = logging.INFO
TRACE_LOG_FILE = "execution_trace.log"
```

### Summary Report Configuration

**Location**: `response/log_summary.py`

```python
# Configure debugging summary report
SUMMARY_MAX_ITEMS = 10         # Limit items shown
SUMMARY_MAX_LENGTH = 500       # Truncate long outputs
INCLUDE_CONSTRAINT_DETAILS = True
INCLUDE_STEP_DETAILS = True
```

## API Configuration

### TMDB API Settings

```python
# Base configuration
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('TMDB_API_KEY')}",
    "Content-Type": "application/json"
}

# Rate limiting
TMDB_REQUESTS_PER_SECOND = 40  # TMDB API limit
REQUEST_DELAY = 0.025          # Delay between requests
```

**Request Configuration**:
```python
# Timeout settings
CONNECT_TIMEOUT = 10           # Connection timeout
READ_TIMEOUT = 30             # Read timeout
TOTAL_TIMEOUT = 45            # Total timeout

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]     # Seconds between retries
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
```

### OpenAI API Settings

```python
# Model configuration
OPENAI_CONFIG = {
    "model": "gpt-3.5-turbo",
    "max_tokens": 1000,
    "temperature": 0.1,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Request settings
OPENAI_TIMEOUT = 30
OPENAI_MAX_RETRIES = 3
```

## Advanced Configuration

### Custom Constraint Types

Add new constraint types by extending `core/model/constraint.py`:

```python
CUSTOM_CONSTRAINTS = {
    "custom_rating_range": {
        "type": "numeric_range",
        "validation": lambda x: 0 <= x <= 10
    },
    "custom_decade": {
        "type": "time_period", 
        "validation": lambda x: 1900 <= x <= 2030
    }
}
```

### Custom Response Formatters

Register new formatters in `core/formatting/registry.py`:

```python
CUSTOM_FORMATTERS = {
    "custom_format": custom_format_function,
    "json_export": json_export_formatter,
    "csv_export": csv_export_formatter
}
```

### Extension Points

**Custom Validators**:
```python
# Add in core/validation/
def custom_validator(result, constraints):
    # Custom validation logic
    return is_valid, validation_details
```

**Custom Extractors**:
```python
# Add in nlp/
def custom_entity_extractor(query):
    # Custom entity extraction
    return entities, intents
```

## Environment-Specific Configurations

### Development Configuration

```env
# .env.development
DEBUG_MODE=true
LOG_LEVEL=DEBUG
ENABLE_TRACE_LOGGING=true
CACHE_DISABLED=true
OPENAI_MODEL=gpt-3.5-turbo
```

### Production Configuration

```env
# .env.production
DEBUG_MODE=false
LOG_LEVEL=WARNING
ENABLE_TRACE_LOGGING=false
CACHE_ENABLED=true
OPENAI_MODEL=gpt-4
RATE_LIMIT_ENABLED=true
```

### Testing Configuration

```env
# .env.testing
DEBUG_MODE=true
USE_MOCK_APIs=true
OPENAI_MODEL=gpt-3.5-turbo
TMDB_API_RATE_LIMIT=disabled
LOG_LEVEL=DEBUG
```

## Troubleshooting Configuration

### Common Configuration Issues

**API Key Problems**:
- Verify `.env` file is in project root
- Check for extra spaces or quotes in API keys
- Ensure TMDB key has proper permissions

**Memory Issues**:
- Reduce `EMBEDDING_BATCH_SIZE`
- Lower `MAX_SEARCH_RESULTS`
- Disable caching temporarily

**Performance Issues**:
- Enable caching
- Use smaller models
- Reduce `MAX_CONCURRENT_REQUESTS`

### Configuration Validation

Add to your startup script:
```python
def validate_configuration():
    required_env = ["TMDB_API_KEY", "OPENAI_API_KEY"]
    for var in required_env:
        if not os.getenv(var):
            raise ValueError(f"Required environment variable {var} not set")

validate_configuration()
```

---

**Note**: After changing configuration, restart the application for changes to take effect. Some configuration changes may require clearing the ChromaDB cache.