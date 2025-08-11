# Advanced Usage Guide

This guide covers advanced features, customization options, and power-user techniques for TMDBGPT.

## 🎆 August 2025 Status Update

TMDBGPT has undergone major improvements with significantly enhanced success rates and new capabilities:

### Query Success Matrix

| Query Type | Examples | Performance | Status |
|------------|----------|-------------|--------|
| **TV Role Queries** | "Who starred in Breaking Bad?", "Who created The Office?" | **Excellent** | ✅ **Working Well** |
| **Movie Role Queries** | "Who wrote Inception?", "Who composed Interstellar?" | **Excellent** | ✅ **Working Well** |
| **TV Count Queries** | "How many seasons does Breaking Bad have?" | **Excellent** | ✅ **Working Well** |
| **Fact Queries** | "How long is Titanic?", "What genre is The Matrix?" | **Good** | ✅ **Working Well** |
| **Single Entity Info** | "Tell me about Inception", "Who is Christopher Nolan?" | **Excellent** | ✅ **Working Well** |
| **Multi-Entity Constraints** | "Movies by Spielberg starring Tom Hanks" | **Good*** | ✅ **Recent Improvements** |
| **Company/Studio Queries** | "Movies by Marvel Studios", "HBO shows" | **Improving** | 🔧 **Under Development** |
| **Complex Triple Constraints** | "2010s sci-fi by Nolan with Hans Zimmer" | **Limited** | 🔧 **Under Development** |

*Multi-entity constraint performance is based on limited recent testing. Results vary by constraint complexity, entity recognition accuracy, and TMDB data availability.

### What Works Excellently Now

**TV Show Role Queries** -
```bash
# TV Creators and Showrunners
"Who created Breaking Bad?"        # → Vince Gilligan, etc.
"Who created The Office?"          # → Greg Daniels, Ricky Gervais
"Who created Game of Thrones?"     # → David Benioff, D.B. Weiss

# TV Cast Members
"Who starred in Breaking Bad?"     # → Bryan Cranston, Aaron Paul, etc.
"Who starred in The Office?"       # → Steve Carell, John Krasinski, etc.
"Who starred in Friends?"          # → Jennifer Aniston, Courteney Cox, etc.

# TV Writers and Producers
"Who wrote Breaking Bad?"          # → Writing team/creators
"Who produced Game of Thrones?"    # → Producers list
```

**Multi-Entity Constraint Queries** - Excellent performance on dual-constraint queries:
```bash
# Actor + Director Combinations  
"Movies by Spielberg starring Tom Hanks"  # → Saving Private Ryan, Catch Me If You Can, The Terminal, Bridge of Spies, The Post
"Movies starring Matt Damon written by Ben Affleck"  # → Good Will Hunting, The Last Duel, Air
"Movies starring Leonardo DiCaprio directed by Martin Scorsese"  # → The Departed, Gangs of New York, The Wolf of Wall Street
```

**Movie Role Queries** - Major enhancement covering all crew roles:
```bash
# Directors and Writers
"Who directed Inception?"          # → Christopher Nolan
"Who wrote Pulp Fiction?"          # → Quentin Tarantino
"Who wrote The Dark Knight?"       # → Jonathan Nolan, Christopher Nolan

# Composers and Producers
"Who composed Interstellar?"       # → Hans Zimmer
"Who composed The Dark Knight?"    # → Hans Zimmer, James Newton Howard
"Who produced The Godfather?"      # → Albert S. Ruddy

# Cast Information
"Who starred in Inception?"        # → Leonardo DiCaprio, Marion Cotillard, etc.
"Who starred in The Godfather?"    # → Marlon Brando, Al Pacino, etc.
```

**TV Show Attributes** 
```bash
# Season and Episode Counts
"How many seasons does Breaking Bad have?"      # → 5 seasons
"How many episodes does The Office have?"       # → 201 episodes across 9 seasons  
"How many seasons does Game of Thrones have?"   # → 8 seasons
"How many episodes does Friends have?"          # → 236 episodes across 10 seasons
```

**Enhanced Fact Queries** 
```bash
# Movie Technical Details
"How long is Titanic?"             # → 194 minutes
"What genre is The Matrix?"        # → Action, Science Fiction
"What was Avengers budget?"        # → $220,000,000
"What year was Blade Runner released?" # → 1982

# TV Show Information
"When did Breaking Bad first air?" # → 2008
"When did The Office first air?"   # → 2005
```

## Advanced Query Techniques

### ✅ Multi-Entity Constraint Queries (Working Well - Recent Testing Shows Strong Performance)

TMDBGPT handles sophisticated queries involving multiple constraints. Recent testing shows excellent performance on actor+director combinations, though results vary by constraint complexity:

#### Triple Constraint Queries
```bash
# Person + Genre + Time Period
"Science fiction movies directed by Ridley Scott from the 1980s"

# Multiple People + Company
"Marvel movies with both Chris Evans and Robert Downey Jr"

# Network + Genre + Rating
"HBO crime dramas with ratings above 8.0"
```

#### Role-Specific Multi-Person Queries
```bash
# Different roles for different people
"Movies written by Charlie Kaufman and directed by Spike Jonze"

# Same person, different roles
"Movies where Ben Affleck was both actor and director"

# Multiple people, same role
"Movies directed by both the Coen Brothers"
```

### Query Pattern Optimization

#### Using Semantic Variations

TMDBGPT understands various phrasings:

```bash
# All equivalent queries:
"Movies directed by Christopher Nolan"
"Films by Christopher Nolan" 
"Christopher Nolan's filmography"
"What movies has Christopher Nolan directed?"
```

#### Leveraging Constraint Relaxation

Structure queries to take advantage of automatic fallback:

```bash
# Primary query with fallback potential
"Thrillers directed by David Fincher"

# System will progressively relax:
# 1. Remove director constraint (show all thrillers)
# 2. Keep thriller genre (core constraint)
# 3. Fallback to similar genres if needed
```

## Advanced Features

### Debug Mode Analysis

Enable comprehensive debugging for query analysis:

```python
# In app.py
DEBUG_MODE = True
```

#### Understanding Debug Output

**Entity Resolution Analysis**:
```
🧾 Extracted Entities:
   - Christopher Nolan (person, role=director) → 525
   - Leonardo DiCaprio (person, role=actor) → 6193
```

**Constraint Tree Evaluation**:
```
📐 Constraint Tree:
ConstraintGroup([
  Constraint(with_people, [525, 6193], AND),
  Constraint(person_roles, {525: director, 6193: actor}, AND)
], logic=AND)
```

**Execution Trace**:
```
🧭 Completed Steps (8):
   - step_extract_entities
   - step_resolve_person_525
   - step_resolve_person_6193
   - step_director_525_movie
   - step_actor_6193_movie
   - step_discover_intersection
   - step_validate_movie_12345
   - step_validate_movie_67890
```

### Performance Optimization

#### Query Structure for Speed

**Faster Queries**:
- Be specific with names: "Christopher Nolan" vs "Nolan"
- Use common entity types: people, genres, years
- Avoid overly complex multi-layered constraints

**Slower Queries**:
- Vague references: "that director who made Inception"
- Unusual entity combinations
- Very broad date ranges

#### Caching Strategies

Enable aggressive caching for repeated queries:

```python
# In configuration
CACHE_ENTITY_RESOLUTIONS = True
CACHE_API_RESPONSES = True
EMBEDDING_CACHE_SIZE = 2000
```

### 🎆 Technical Architecture Improvements (August 2025)

#### Symbol-Free Routing System ✨ NEW

**Technical Details**:
- **Detection**: Queries like "Who starred in Breaking Bad?" identified as single-entity fact queries
- **Routing**: Direct routing to `/tv/{id}?append_to_response=credits` instead of complex constraint planning
- **Benefits**: Faster execution, more accurate results, comprehensive role data

**Examples of Symbol-Free Routing**:
```python
# Query: "Who created Breaking Bad?"
# Old Path: Complex constraint planning → Often failed
# New Path: Direct TV lookup → /tv/1396?append_to_response=credits → Success

# Query: "Who wrote Inception?" 
# Old Path: Generic discovery → Incomplete data
# New Path: Direct movie lookup → /movie/27205?append_to_response=credits → Complete crew
```

#### Enhanced Credits API Integration 
**Complete Role Support**: All TMDB crew roles now extracted and supported:

**Movie Roles Supported**:
- Directors, Writers (screenplay/story), Composers, Producers
- Executive Producers, Original Music Composers
- Main Cast (top 5 for performance)

**TV Show Roles Supported**:
- Creators, Showrunners, Executive Producers
- Writers, Producers, Co-Executive Producers  
- Main Cast (top 5 for performance)

**Technical Implementation**:
```python
# Enhanced extraction logic
def _extract_movie_details(json_data, endpoint):
    # Extract all crew roles from credits API
    for crew_member in json_data["credits"].get("crew", []):
        job = crew_member.get("job", "").lower()
        if job == "director":
            directors.append(name)
        elif job in ["writer", "screenplay", "story"]:
            writers.append(name)
        elif job in ["original music composer", "composer"]:
            composers.append(name)
        # ... additional role extraction
```

#### Intent Correction System 

**Problem Solved**: LLM sometimes misclassifies TV shows as movies or vice versa, causing routing failures.

**Solution**: Automatic intent correction for role-based queries:
```python
# Example correction logic
if tv_entities and "details.movie" in intents:
    # Correct movie intent to TV for shows
    intents.remove("details.movie")
    intents.append("details.tv")
```

**Impact**: Eliminates classification-based failures for TV role queries.

#### Enhanced Fact Extraction Pipeline 

**Keyword-Based Detection**: Intelligent fact type detection based on query keywords:

```python
# New fact detection logic
is_runtime_question = any(keyword in query_text for keyword in 
    ["long", "runtime", "duration", "minutes", "hours"])
is_genre_question = any(keyword in query_text for keyword in 
    ["genre", "type of", "kind of", "category"])
is_director_question = any(keyword in query_text for keyword in 
    ["direct", "director", "directed"])
```

**Field Prioritization**: Query-specific field extraction prioritization:
- Runtime queries → Extract and return runtime field first
- Genre queries → Extract and format genre list
- Role queries → Extract comprehensive crew/cast information

### Custom Query Patterns

#### Using Natural Language Patterns

**Enhanced TV Role Patterns**
```bash
# Creator/Showrunner Queries
"Who created [Show]?"
"Who developed [Show]?"
"[Show] was created by who?"

# Cast and Crew Queries  
"Who starred in [Show]?"
"Cast of [Show]"
"Who wrote [Show]?"
"Who produced [Show]?"
```

**Enhanced Movie Role Patterns**
```bash
# Comprehensive Crew Queries
"Who wrote [Movie]?"
"Who composed [Movie]?"
"Who produced [Movie]?"
"Music by whom in [Movie]?"
"[Movie] director?"
```

**Enhanced Fact Patterns**
```bash
# Technical Details
"How long is [Movie]?"
"Runtime of [Movie]"
"What genre is [Movie]?"
"Budget of [Movie]?"

# TV Attributes
"How many seasons [Show]?"
"Episode count [Show]"
```

**Statistical Queries**:
```bash
"How many movies has Quentin Tarantino directed?"
"How many seasons does Breaking Bad have?" ✨
"How many episodes does The Office have?" ✨
```

## Advanced Configuration

### Custom Response Formatting

#### Creating Custom Formatters

```python
# In core/formatting/custom_formatters.py
def detailed_movie_formatter(state):
    """Custom formatter with extended movie details."""
    formatted_results = []
    
    for movie in state.responses:
        details = f"""
🎬 {movie.get('title', 'Unknown')} ({movie.get('release_date', 'TBD')[:4]})
   📊 Rating: {movie.get('vote_average', 'N/A')}/10
   🎭 Director: {get_director(movie)}
   ⭐ Cast: {get_main_cast(movie)}
   📝 Overview: {movie.get('overview', 'No overview available')[:200]}...
   🏢 Studio: {get_production_companies(movie)}
   💰 Budget: ${movie.get('budget', 0):,}
   """
        formatted_results.append(details.strip())
    
    return formatted_results
```

#### Registering Custom Formatters

```python
# In core/formatting/registry.py
from .custom_formatters import detailed_movie_formatter

RESPONSE_RENDERERS = {
    "summary": default_summary_formatter,
    "detailed": detailed_movie_formatter,  # Add custom formatter
    "ranked_list": ranked_list_formatter,
}
```

### Advanced Entity Resolution

#### Custom Entity Types

```python
# In core/entity/custom_resolvers.py
class CustomEntityResolver(TMDBEntityResolver):
    """Extended resolver with custom entity types."""
    
    def resolve_franchise(self, name: str) -> Optional[dict]:
        """Resolve franchise/series names."""
        # Custom logic for franchise resolution
        pass
    
    def resolve_award_category(self, category: str) -> Optional[dict]:
        """Resolve award categories to filter criteria."""
        award_mappings = {
            "oscar winner": {"vote_average.gte": 7.0},
            "cannes winner": {"with_companies": [cannes_company_id]},
        }
        return award_mappings.get(category.lower())
```

#### Enhanced Fuzzy Matching

```python
# Custom fuzzy matching with domain knowledge
def enhanced_person_matching(query_name: str, candidates: List[dict]) -> dict:
    """Enhanced person matching with nickname handling."""
    
    # Nickname mappings
    nicknames = {
        "leo": "leonardo dicaprio",
        "rdj": "robert downey jr",
        "the rock": "dwayne johnson",
    }
    
    normalized_query = nicknames.get(query_name.lower(), query_name)
    return fuzzy_match(normalized_query, candidates)
```

### Advanced Constraint Handling

#### Custom Constraint Types

```python
# In core/model/custom_constraints.py
class RatingRangeConstraint(Constraint):
    """Custom constraint for rating ranges."""
    
    def __init__(self, min_rating: float, max_rating: float):
        super().__init__("vote_average", (min_rating, max_rating))
        
    def evaluate(self, data: dict) -> bool:
        rating = data.get("vote_average", 0)
        min_val, max_val = self.value
        return min_val <= rating <= max_val

class DecadeConstraint(Constraint):
    """Custom constraint for decades."""
    
    def __init__(self, decade: int):
        start_year = decade
        end_year = decade + 9
        super().__init__("release_year_range", (start_year, end_year))
        
    def evaluate(self, data: dict) -> bool:
        release_date = data.get("release_date", "")
        if not release_date:
            return False
        year = int(release_date[:4])
        start_year, end_year = self.value
        return start_year <= year <= end_year
```

#### Complex Constraint Logic

```python
# Advanced constraint combinations
def build_complex_constraints(entities):
    """Build sophisticated constraint combinations."""
    
    # OR logic for alternative people
    alternative_directors = ConstraintGroup([
        Constraint("with_people", [nolan_id]),
        Constraint("with_people", [villeneuve_id]),
        Constraint("with_people", [scott_id])
    ], logic="OR")
    
    # AND logic for required elements
    required_elements = ConstraintGroup([
        Constraint("with_genres", [sci_fi_id]),
        RatingRangeConstraint(7.0, 10.0),
        DecadeConstraint(2010)
    ], logic="AND")
    
    # Combined constraint tree
    return ConstraintGroup([
        alternative_directors,
        required_elements
    ], logic="AND")
```

## Integration and Automation

### Programmatic Usage

#### Direct API Usage

```python
from core.execution_state import AppState
from app import build_app_graph

def query_tmdbgpt(user_query: str) -> dict:
    """Programmatic interface to TMDBGPT."""
    graph = build_app_graph()
    result = graph.invoke({"input": user_query})
    return result

# Usage
results = query_tmdbgpt("Movies directed by Christopher Nolan")
for movie in results.get("responses", []):
    print(movie)
```

#### Batch Query Processing

```python
def process_batch_queries(queries: List[str]) -> List[dict]:
    """Process multiple queries efficiently."""
    graph = build_app_graph()
    results = []
    
    for query in queries:
        try:
            result = graph.invoke({"input": query})
            results.append({
                "query": query,
                "success": True,
                "results": result.get("responses", [])
            })
        except Exception as e:
            results.append({
                "query": query, 
                "success": False,
                "error": str(e)
            })
    
    return results
```

### Custom Workflows

#### Query Preprocessing

```python
def preprocess_query(query: str) -> str:
    """Custom query preprocessing."""
    
    # Expand abbreviations
    abbreviations = {
        "MCU": "Marvel Cinematic Universe",
        "DCU": "DC Universe",
        "SW": "Star Wars"
    }
    
    for abbr, full in abbreviations.items():
        query = query.replace(abbr, full)
    
    # Fix common typos
    typo_corrections = {
        "Cristopher": "Christopher",
        "DiCaprio": "DiCaprio"
    }
    
    for typo, correction in typo_corrections.items():
        query = query.replace(typo, correction)
    
    return query
```

#### Result Postprocessing

```python
def postprocess_results(results: List[dict]) -> List[dict]:
    """Custom result postprocessing."""
    
    # Add custom scoring
    for result in results:
        result["custom_score"] = calculate_custom_score(result)
    
    # Sort by custom criteria
    results.sort(key=lambda x: x["custom_score"], reverse=True)
    
    # Add metadata
    for i, result in enumerate(results):
        result["rank"] = i + 1
        result["recommendation_reason"] = get_recommendation_reason(result)
    
    return results
```

## Performance Monitoring

### Query Performance Analysis

#### Timing Analysis

```python
import time
from functools import wraps

def time_execution(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        print(f"{func.__name__} took {execution_time:.2f} seconds")
        return result
    return wrapper

# Apply to graph functions
@time_execution
def timed_query_processing(query: str):
    graph = build_app_graph()
    return graph.invoke({"input": query})
```

#### Memory Usage Monitoring

```python
import psutil
import gc

def monitor_memory_usage():
    """Monitor memory usage during query processing."""
    process = psutil.Process()
    
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"Memory percent: {process.memory_percent():.1f}%")
    
    # Force garbage collection
    gc.collect()
```

### Optimization Strategies

#### Query Caching System

```python
from functools import lru_cache
import hashlib

class QueryCache:
    """Advanced query caching system."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[dict]:
        """Get cached result."""
        key = self.get_cache_key(query)
        return self.cache.get(key)
    
    def set(self, query: str, result: dict):
        """Cache result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self.get_cache_key(query)
        self.cache[key] = result
```

## Advanced Troubleshooting

### Custom Debug Output

#### Enhanced Debugging

```python
def debug_query_processing(query: str):
    """Comprehensive query debugging."""
    
    print(f"🔍 Analyzing query: '{query}'")
    
    # Step-by-step debugging
    graph = build_app_graph()
    
    # Custom state tracking
    debug_state = {"steps": [], "timings": {}}
    
    def debug_wrapper(original_func):
        def wrapper(state):
            start_time = time.time()
            result = original_func(state)
            execution_time = time.time() - start_time
            
            debug_state["steps"].append(original_func.__name__)
            debug_state["timings"][original_func.__name__] = execution_time
            
            return result
        return wrapper
    
    # Apply debug wrappers
    # Process with enhanced debugging
    result = graph.invoke({"input": query})
    
    # Output debug information
    print(f"📊 Execution Summary:")
    for step, timing in debug_state["timings"].items():
        print(f"   {step}: {timing:.3f}s")
    
    return result
```

### Performance Profiling

#### Detailed Performance Analysis

```python
import cProfile
import pstats

def profile_query_processing(query: str):
    """Profile query processing performance."""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Process query
    graph = build_app_graph()
    result = graph.invoke({"input": query})
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

## Future Extensions

### Plugin Architecture

#### Custom Plugin Interface

```python
class TMDBGPTPlugin:
    """Base class for TMDBGPT plugins."""
    
    def __init__(self, name: str):
        self.name = name
    
    def preprocess_query(self, query: str) -> str:
        """Hook for query preprocessing."""
        return query
    
    def postprocess_results(self, results: List[dict]) -> List[dict]:
        """Hook for result postprocessing."""
        return results
    
    def custom_entity_resolution(self, entity: dict) -> Optional[dict]:
        """Hook for custom entity resolution."""
        return None
```

#### Example Plugin

```python
class IMDbIntegrationPlugin(TMDBGPTPlugin):
    """Plugin to integrate IMDb data."""
    
    def __init__(self):
        super().__init__("IMDb Integration")
    
    def postprocess_results(self, results: List[dict]) -> List[dict]:
        """Add IMDb ratings to results."""
        for result in results:
            imdb_id = result.get("imdb_id")
            if imdb_id:
                imdb_rating = self.fetch_imdb_rating(imdb_id)
                result["imdb_rating"] = imdb_rating
        return results
```

---

This advanced usage guide provides the foundation for power users and developers to extend and customize TMDBGPT for sophisticated use cases.