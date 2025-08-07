# Contributing Guidelines

Thank you for your interest in contributing to TMDBGPT! This guide will help you understand our development process, coding standards, and how to contribute effectively.

## Getting Started

### Prerequisites

Before contributing, ensure you have:
- Python 3.8+ installed
- Git configured with your GitHub account
- TMDB and OpenAI API keys for testing
- Familiarity with the [Architecture Documentation](architecture.md)

### Setting Up Development Environment

1. **Fork and Clone**
   ```bash
   git fork https://github.com/FinnMacCumail/tmdbGPT.git
   git clone https://github.com/your-username/tmdbGPT.git
   cd tmdbGPT
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

4. **Verify Setup**
   ```bash
   python app.py
   # Test with a simple query
   ```

## Development Workflow

We follow a **feature branch workflow** as documented in our [Development Workflow](DEVELOPMENT_WORKFLOW.md).

### Creating a Feature

1. **Start from Main**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-description
   ```

3. **Develop Your Feature**
   - Add temporary debug statements as needed during development
   - Test thoroughly with various query types
   - Document any new functionality

4. **Pre-Merge Cleanup**
   - Remove all temporary debug statements
   - Ensure 🧠 DEBUGGING SUMMARY REPORT functionality remains intact
   - Run tests and verify functionality
   - Update documentation if needed

5. **Merge and Push**
   ```bash
   git checkout main
   git merge feature/your-feature-description
   git push origin main
   git branch -d feature/your-feature-description
   ```

## Code Standards

### Python Code Style

We follow **PEP 8** with these specific guidelines:

**Imports**
```python
# Standard library imports first
import os
import time
from typing import List, Dict, Any

# Third-party imports
import requests
from pydantic import BaseModel

# Local imports last  
from core.execution_state import AppState
from core.model.constraint import Constraint
```

**Function Documentation**
```python
def resolve_entity(name: str, entity_type: str) -> Optional[dict]:
    """
    Resolve entity name to TMDB ID.
    
    Args:
        name: Entity name to resolve
        entity_type: Type of entity ("person", "movie", "tv")
        
    Returns:
        Optional[dict]: Entity details with ID if found, None otherwise
        
    Raises:
        EntityResolutionError: If API request fails
    """
```

**Class Structure**
```python
class TMDBEntityResolver:
    """Resolves entity names to TMDB IDs with caching and validation."""
    
    def __init__(self, api_key: str, headers: dict):
        """Initialize resolver with API credentials."""
        self.api_key = api_key
        self.headers = headers
        self._cache = {}
    
    def resolve_person(self, name: str) -> Optional[dict]:
        """Resolve person name to TMDB person data."""
        # Implementation here
```

### Comment Guidelines

**DO NOT** add comments unless they provide significant value:

❌ **Bad Comments**:
```python
# Increment counter
counter += 1

# Check if result is not None
if result is not None:
```

✅ **Good Comments**:
```python
# TMDB API returns OR logic for with_people, but query expects AND
# Post-validate to ensure all people are actually in the movie
validated_results = self._validate_all_people_present(results, people_ids)

# ChromaDB vector search returns similarity scores 0-1
# We use 0.7 threshold based on empirical testing with movie queries
if match_score >= 0.7:
```

**Code Structure Comments**:
```python
# 🧠 Entity & Intent Extraction Step
# Converts the user's raw query (state.input) into structured semantic fields
def extract_entities(state: AppState) -> AppState:
```

### Error Handling

**Use Specific Exceptions**:
```python
# Good
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except requests.Timeout:
    logger.error(f"Request timeout for {url}")
    return None
except requests.HTTPError as e:
    logger.error(f"HTTP error {e.response.status_code}: {e}")
    return None
```

**Graceful Degradation**:
```python
try:
    advanced_result = complex_processing(data)
    return advanced_result
except ProcessingError:
    logger.warning("Advanced processing failed, using fallback")
    return simple_processing(data)
```

### Testing Requirements

**Unit Tests**: Required for all new functionality
```python
def test_resolve_person_success():
    """Test successful person resolution."""
    resolver = TMDBEntityResolver(api_key="test", headers={})
    result = resolver.resolve_person("Leonardo DiCaprio")
    
    assert result is not None
    assert result["name"] == "Leonardo DiCaprio"
    assert "id" in result
```

**Integration Tests**: Required for end-to-end flows
```python
def test_full_query_processing():
    """Test complete query processing pipeline."""
    graph = build_app_graph()
    result = graph.invoke({"input": "Movies directed by Christopher Nolan"})
    
    assert "responses" in result
    assert len(result["responses"]) > 0
```

**Test Organization**:
- Place tests in `unit_tests/` directory
- Use descriptive test names
- Include both positive and negative test cases
- Test edge cases and error conditions

## Feature Development Guidelines

### New Query Types

When adding support for new query types:

1. **Update Entity Extraction** (`core/llm/extractor.py`)
   - Add new entity types
   - Update intent classification
   - Add validation logic

2. **Update Constraint System** (`core/model/constraint.py`)
   - Add new constraint types
   - Implement evaluation logic
   - Add validation rules

3. **Add Validators** (`core/validation/`)
   - Implement specific validation logic
   - Add to validator registry
   - Test with various inputs

4. **Update Response Formatting** (`core/formatting/`)
   - Add new response templates
   - Update formatter registry
   - Ensure consistent output format

### New Data Sources

To integrate new data sources:

1. **Create Resolver** (`core/entity/`)
   ```python
   class NewDataSourceResolver:
       def resolve_entities(self, entities: List[dict]) -> List[dict]:
           # Implementation
   ```

2. **Update Semantic Search** (`core/embeddings/`)
   - Add new endpoint definitions
   - Update embedding generation
   - Test semantic matching

3. **Add API Handlers** (`core/execution/`)
   - Implement request/response handling
   - Add error handling and retries
   - Update execution orchestrator

### Performance Improvements

When optimizing performance:

1. **Measure First**: Use profiling to identify bottlenecks
2. **Cache Strategically**: Cache expensive operations
3. **Optimize Requests**: Use connection pooling and batching
4. **Monitor Memory**: Avoid memory leaks in long-running processes

## Documentation Requirements

### Code Documentation

**Module-Level Documentation**:
```python
"""
Entity resolution module for TMDBGPT.

This module provides functionality to resolve natural language entity
names (people, movies, TV shows) to their corresponding TMDB IDs using
fuzzy matching and caching strategies.
"""
```

**Function Documentation**: Required for all public functions
**Class Documentation**: Required for all classes
**Complex Logic**: Comment non-obvious algorithms

### User Documentation

When adding user-facing features:
- Update relevant documentation in `docs/`
- Add examples to User Guide
- Update README if it affects basic usage
- Consider adding to troubleshooting guide

## Testing Guidelines

### Test Structure

```python
class TestEntityResolver:
    """Test suite for TMDBEntityResolver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = TMDBEntityResolver("test_key", {})
    
    def test_resolve_person_success(self):
        """Test successful person resolution."""
        # Test implementation
    
    def test_resolve_person_not_found(self):
        """Test person not found scenario."""
        # Test implementation
    
    def test_resolve_person_api_error(self):
        """Test API error handling.""" 
        # Test implementation
```

### Test Coverage

Aim for high test coverage:
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete user workflows
- **Error Cases**: Test error handling and edge cases

### Running Tests

```bash
# Run all tests
python -m pytest unit_tests/

# Run specific test file
python -m pytest unit_tests/test_entity_resolver.py

# Run with coverage
python -m pytest --cov=core unit_tests/
```

## Pull Request Process

### Before Submitting

1. **Test Thoroughly**
   - Run all existing tests
   - Add tests for new functionality
   - Test with various query types
   - Verify 🧠 DEBUGGING SUMMARY REPORT still works

2. **Clean Up Code**
   - Remove temporary debug statements
   - Remove commented-out code
   - Ensure proper formatting
   - Update documentation

3. **Verify Integration**
   - Test with both DEBUG_MODE settings
   - Verify no regressions
   - Test error handling

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Edge cases tested

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No temporary debug statements
- [ ] Tests pass
```

### Review Process

1. **Automated Checks**: All tests must pass
2. **Code Review**: At least one maintainer review required
3. **Documentation Review**: Check for documentation updates
4. **Manual Testing**: Verify functionality works as expected

## Code Review Guidelines

### For Contributors

**Self-Review Checklist**:
- [ ] Code is clean and well-documented
- [ ] Tests cover new functionality
- [ ] No temporary or debug code included
- [ ] Documentation is updated
- [ ] Performance impact considered

**Responding to Feedback**:
- Address all review comments
- Explain design decisions if needed
- Be open to suggestions and improvements
- Update tests based on feedback

### For Reviewers

**Review Focus Areas**:
- Code correctness and logic
- Test coverage and quality
- Documentation completeness
- Performance implications
- Security considerations

**Providing Feedback**:
- Be constructive and specific
- Suggest improvements where possible
- Acknowledge good practices
- Test the changes locally if needed

## Issue Reporting

### Bug Reports

Include in your bug report:
- **Python Version**: `python --version`
- **Operating System**: OS and version
- **Query That Failed**: Exact query text
- **Expected Behavior**: What should have happened
- **Actual Behavior**: What actually happened
- **Debug Output**: Set `DEBUG_MODE = True` and include 🧠 DEBUGGING SUMMARY REPORT
- **Steps to Reproduce**: Minimal steps to reproduce the issue

### Feature Requests

Include in your feature request:
- **Problem Statement**: What problem does this solve?
- **Proposed Solution**: How should it work?
- **Use Cases**: Example scenarios where this would be useful
- **Alternatives Considered**: Other solutions you've considered

## Getting Help

### Documentation Resources
- [User Guide](user-guide.md) - Using TMDBGPT
- [Architecture Documentation](architecture.md) - System design
- [API Reference](api-reference.md) - Developer APIs
- [Development Workflow](DEVELOPMENT_WORKFLOW.md) - Workflow details

### Community Support
- **GitHub Issues**: Technical questions and bug reports
- **GitHub Discussions**: General questions and feature discussions

### Maintainer Contact
- Create an issue for technical questions
- Tag maintainers in discussions for urgent matters

---

**Thank you for contributing to TMDBGPT!** 🎬 Your contributions help make movie and TV discovery more natural and accessible for everyone.