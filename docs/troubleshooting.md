# Troubleshooting Guide

This guide helps resolve common issues when installing, configuring, or using TMDBGPT.

## Installation Issues

### Python and Dependencies

#### "Python version not supported"

**Error**: `Python 3.x.x is not supported`

**Solution**:
```bash
# Check Python version
python --version

# Install Python 3.8+ if needed
# On Ubuntu/Debian:
sudo apt update && sudo apt install python3.10

# On macOS with Homebrew:
brew install python@3.10

# On Windows: Download from python.org
```

#### "ModuleNotFoundError" during installation

**Error**: `ModuleNotFoundError: No module named 'xyz'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Upgrade pip first
pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt

# If still failing, try installing individually
pip install requests openai chromadb sentence-transformers spacy
```

#### "Permission denied" errors

**Error**: Permission errors during installation

**Solution**:
```bash
# Don't use sudo with pip in virtual environment
# Instead, ensure virtual environment is properly activated

# If system-wide installation needed (not recommended):
pip install --user -r requirements.txt
```

### SpaCy Model Issues

#### "Can't find model 'en_core_web_sm'"

**Error**: `OSError: [E050] Can't find model 'en_core_web_sm'`

**Solution**:
```bash
# Download the English model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import spacy; spacy.load('en_core_web_sm')"

# If download fails, try direct link:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
```

#### SpaCy version conflicts

**Error**: SpaCy model compatibility issues

**Solution**:
```bash
# Check SpaCy version
python -c "import spacy; print(spacy.__version__)"

# Download compatible model
python -m spacy download en_core_web_sm

# If problems persist, reinstall SpaCy
pip uninstall spacy
pip install spacy==3.7.4
python -m spacy download en_core_web_sm
```

## Configuration Issues

### API Key Problems

#### "Invalid TMDB API Key"

**Error**: `401 Unauthorized` or API key validation errors

**Solution**:
1. **Verify API Key Format**:
   ```bash
   # TMDB Bearer token should start with 'eyJ'
   echo $TMDB_API_KEY
   ```

2. **Check .env File Location**:
   ```bash
   # Ensure .env is in project root
   ls -la .env
   ```

3. **Verify .env Format**:
   ```env
   TMDB_API_KEY=eyJhbGciOiJIUzI1NiJ9...
   OPENAI_API_KEY=sk-proj-...
   ```

4. **Test API Key**:
   ```bash
   curl -H "Authorization: Bearer YOUR_TMDB_KEY" \
        "https://api.themoviedb.org/3/configuration"
   ```

#### "OpenAI API Key Invalid"

**Error**: OpenAI authentication errors

**Solution**:
1. **Check API Key Format**:
   ```bash
   # OpenAI key should start with 'sk-'
   echo $OPENAI_API_KEY
   ```

2. **Verify API Credits**:
   - Check OpenAI dashboard for remaining credits
   - Ensure billing is set up if required

3. **Test OpenAI Connection**:
   ```python
   import openai
   client = openai.OpenAI()
   response = client.models.list()
   print("OpenAI connection successful")
   ```

### Environment Variable Issues

#### "Environment variable not found"

**Error**: Required environment variables missing

**Solution**:
1. **Check .env File Loading**:
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   print(os.getenv('TMDB_API_KEY'))  # Should not be None
   ```

2. **Verify .env File Contents**:
   ```bash
   cat .env
   # Should show your API keys without quotes
   ```

3. **Check File Permissions**:
   ```bash
   ls -la .env
   # Should be readable by current user
   ```

## Runtime Issues

### ChromaDB Problems

#### "ChromaDB initialization failed"

**Error**: ChromaDB database errors

**Solution**:
1. **Delete and Reinitialize Database**:
   ```bash
   rm -rf chroma_db/
   python app.py  # Will recreate database
   ```

2. **Check Disk Space**:
   ```bash
   df -h .  # Ensure sufficient disk space
   ```

3. **Permissions Check**:
   ```bash
   ls -la chroma_db/
   # Ensure directory is writable
   ```

#### "Collection not found" errors

**Error**: ChromaDB collection errors

**Solution**:
```bash
# Reset ChromaDB completely
rm -rf chroma_db/
python -c "
from core.embeddings.semantic_embed import embed_tmdb_parameters
embed_tmdb_parameters()
print('ChromaDB reinitialized')
"
```

### Memory Issues

#### "Out of memory" errors

**Error**: System runs out of memory during processing

**Solution**:
1. **Reduce Batch Sizes**:
   ```python
   # Edit configuration files to use smaller batches
   EMBEDDING_BATCH_SIZE = 8  # Reduce from default 32
   MAX_SEARCH_RESULTS = 10   # Reduce from default 50
   ```

2. **Use CPU-only Mode**:
   ```python
   # Force CPU usage for sentence transformers
   import torch
   device = "cpu"  # Instead of auto-detection
   ```

3. **Close Other Applications**:
   - Close browsers and other memory-intensive apps
   - Consider using a system with more RAM

#### "CUDA out of memory" (GPU users)

**Error**: GPU memory exhaustion

**Solution**:
```python
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# OR reduce GPU batch sizes
CUDA_BATCH_SIZE = 4  # Very small batch size
```

### Network Issues

#### "Connection timeout" errors

**Error**: API requests timing out

**Solution**:
1. **Check Internet Connection**:
   ```bash
   ping api.themoviedb.org
   ping api.openai.com
   ```

2. **Increase Timeouts**:
   ```python
   # Edit timeout settings in configuration
   REQUEST_TIMEOUT = 60  # Increase from 30 seconds
   ```

3. **Check Firewall/Proxy**:
   - Ensure HTTPS traffic is allowed
   - Configure proxy settings if needed

#### "Rate limit exceeded" errors

**Error**: API rate limiting

**Solution**:
1. **Add Request Delays**:
   ```python
   import time
   time.sleep(0.1)  # Add delay between requests
   ```

2. **Implement Exponential Backoff**:
   ```python
   # Automatic retry with increasing delays
   for retry in range(3):
       try:
           response = make_request()
           break
       except RateLimitError:
           time.sleep(2 ** retry)
   ```

## Query Processing Issues

### No Results Returned

#### "No results found" for valid queries

**Problem**: Query returns empty results despite valid input

**Solution**:
1. **Enable Debug Mode**:
   ```python
   # Set DEBUG_MODE = True in app.py
   DEBUG_MODE = True
   ```

2. **Check Entity Resolution**:
   - Look for "Entity resolution failed" in debug output
   - Try alternative spellings of names
   - Use full names instead of nicknames

3. **Simplify Query**:
   ```bash
   # Instead of complex query
   "Movies directed by Christopher Nolan starring Leonardo DiCaprio from 2010s"
   
   # Try simpler version
   "Movies directed by Christopher Nolan"
   ```

4. **Check Constraint Relaxation**:
   - Debug output shows which constraints were relaxed
   - If all constraints dropped, may indicate data issues

### Incorrect Results

#### "Results don't match query"

**Problem**: Results seem unrelated to the query

**Solution**:
1. **Check Debug Output**:
   ```python
   # Enable debug mode and look for:
   # - Entity extraction results
   # - Constraint evaluation
   # - Role validation results
   ```

2. **Verify Entity Names**:
   ```bash
   # Use exact, full names
   "Christopher Nolan" instead of "Nolan"
   "Leonardo DiCaprio" instead of "Leo"
   ```

3. **Be More Specific**:
   ```bash
   # Add more constraints
   "Movies directed by Christopher Nolan starring Leonardo DiCaprio"
   # Instead of just
   "Nolan DiCaprio movies"
   ```

### Slow Performance

#### "Queries take too long to process"

**Problem**: Very slow response times

**Solution**:
1. **Check Debug Times**:
   - Enable debug mode to see timing information
   - Identify slowest components

2. **Optimize Configuration**:
   ```python
   # Reduce search complexity
   MAX_SEARCH_RESULTS = 10
   ENABLE_DEEP_VALIDATION = False
   ```

3. **Cache Warming**:
   ```bash
   # Run some queries to warm up caches
   # Subsequent queries should be faster
   ```

4. **System Resources**:
   - Ensure adequate RAM available
   - Check CPU usage during queries
   - Consider SSD storage for better I/O

## Application-Specific Issues

### Debug Mode Problems

#### "🧠 DEBUGGING SUMMARY REPORT not showing"

**Problem**: Debug output not appearing when expected

**Solution**:
1. **Check DEBUG_MODE Setting**:
   ```python
   # In app.py, ensure:
   DEBUG_MODE = True
   ```

2. **Verify Import Path**:
   ```python
   # In step_runner.py, check import
   try:
       import app
       if app.DEBUG_MODE:
           log_summary(state)
   ```

3. **Check Log Summary Function**:
   ```python
   # Test log_summary directly
   from response.log_summary import log_summary
   log_summary(mock_state)
   ```

#### "Progress indicators not working"

**Problem**: User-friendly progress messages not showing

**Solution**:
1. **Check DEBUG_MODE Setting**:
   ```python
   # For user-friendly mode:
   DEBUG_MODE = False
   ```

2. **Verify Print Statements**:
   ```python
   # Each graph function should have:
   if not DEBUG_MODE:
       print("🔍 Understanding your question...", flush=True)
   ```

### Import Errors

#### "Cannot import core modules"

**Error**: Import errors for internal modules

**Solution**:
1. **Check Working Directory**:
   ```bash
   # Run from project root directory
   cd /path/to/tmdbGPT
   python app.py
   ```

2. **Verify Python Path**:
   ```bash
   # Add current directory to Python path
   export PYTHONPATH="${PYTHONPATH}:."
   python app.py
   ```

3. **Check File Structure**:
   ```bash
   # Ensure all __init__.py files exist
   find . -name "__init__.py"
   ```

## Performance Optimization

### Memory Optimization

```python
# Reduce memory usage
EMBEDDING_BATCH_SIZE = 8
MAX_CONCURRENT_REQUESTS = 2
CACHE_SIZE = 100
```

### Speed Optimization

```python
# Increase processing speed
USE_GPU = True  # If available
PARALLEL_PROCESSING = True
AGGRESSIVE_CACHING = True
```

### Storage Optimization

```bash
# Clean up old ChromaDB data
rm -rf chroma_db/
# Will be recreated on next run
```

## Getting Additional Help

### Debug Information to Collect

When reporting issues, include:

1. **System Information**:
   ```bash
   python --version
   uname -a  # Linux/macOS
   systeminfo  # Windows
   ```

2. **Environment Setup**:
   ```bash
   pip list | grep -E "(openai|chromadb|spacy|sentence-transformers)"
   ```

3. **Error Output**:
   - Complete error message and traceback
   - Debug output with `DEBUG_MODE = True`
   - Query that caused the issue

4. **Configuration**:
   - Relevant environment variables (without API keys!)
   - Any custom configuration changes

### Useful Commands for Debugging

```bash
# Check Python environment
which python
python -c "import sys; print(sys.path)"

# Test API connections
python -c "import requests; print(requests.get('https://api.themoviedb.org/3/configuration').status_code)"

# Check ChromaDB
python -c "import chromadb; print('ChromaDB OK')"

# Memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

### Where to Get Help

1. **Check Documentation**: Review all relevant documentation first
2. **Search Issues**: Look through existing GitHub issues
3. **Create Issue**: Provide detailed information as described above
4. **Community Support**: Use GitHub Discussions for general questions

---

**Remember**: Most issues are related to configuration, API keys, or environment setup. Start with the basics and work through the solutions systematically.