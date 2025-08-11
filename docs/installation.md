# Installation Guide

This guide will walk you through installing and setting up TMDBGPT on your system.

## Prerequisites

Before installing TMDBGPT, ensure you have:

1. **Python 3.8 or higher** installed on your system
2. **Git** for cloning the repository
3. **TMDB API Key** (free account required)
4. **OpenAI API Key** (paid account required)

## Step 1: Clone the Repository

```bash
git clone https://github.com/FinnMacCumail/tmdbGPT.git
cd tmdbGPT
```

## Step 2: Create Virtual Environment

### On Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Core Dependencies Include:
- **OpenAI**: GPT integration for natural language processing
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: Text embedding models  
- **Requests**: HTTP client for TMDB API calls
- **LangGraph**: State machine orchestration
- **SpaCy**: Natural language processing
- **Python-dotenv**: Environment variable management

## Step 4: Obtain API Keys

### TMDB API Key

1. Create a free account at [The Movie Database (TMDB)](https://www.themoviedb.org/)
2. Go to your [API Settings](https://www.themoviedb.org/settings/api)
3. Request an API key (choose "Developer" if asked)
4. Copy your **API Read Access Token** (starts with `eyJhbGciOiJIUzI1NiJ9...`)

### OpenAI API Key

1. Create an account at [OpenAI Platform](https://platform.openai.com/)
2. Add billing information (required for API access)
3. Go to [API Keys](https://platform.openai.com/api-keys)
4. Create a new API key
5. Copy your API key (starts with `sk-proj-...` or `sk-...`)

**Note**: OpenAI API usage incurs costs. Typical TMDBGPT usage costs $0.10-0.50 per session depending on query complexity.

## Step 5: Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
# Copy the template
cp .env.example .env
```

Edit the `.env` file with your API keys:

```env
# TMDB API Configuration
TMDB_API_KEY=your_tmdb_api_key_here

# OpenAI API Configuration  
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Alternative TMDB API Key format
NON_B_TMDB_API_KEY=your_tmdb_v3_api_key_here
```

**Example `.env` file:**
```env
TMDB_API_KEY=your_tmdb_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Step 6: Initialize ChromaDB Vector Database

**CRITICAL**: ChromaDB must be manually initialized before first use.

### Manual Database Setup (Required)
```bash
# Populate TMDB API endpoint embeddings (54 endpoints)
python core/embeddings/semantic_embed.py

# Populate parameter search embeddings  
python core/embeddings/embed_tmdb_parameters.py
```

### Verify ChromaDB Initialization
```bash
# Check that chroma_db directory was created
ls chroma_db/
# Expected: chroma.sqlite3 and collection subdirectories

# Test semantic search functionality
python -c "from core.embeddings.hybrid_retrieval import retrieve_semantic_matches; print(retrieve_semantic_matches('find movies by actor'))"
```

### ChromaDB Setup Notes
- **First Run**: Sentence transformer model downloads (~90MB) automatically
- **Processing Time**: Initial setup takes 2-3 minutes
- **Storage**: Creates ~20MB of vector database files

## Step 7: Test Installation

Run a quick test to verify everything works:

```bash
python app.py
```

You should see:
```
Ask something (or type 'exit' to quit):
```

Try a simple query:
```
Tell me about Inception
```

If successful, you'll see movie information displayed. Type `exit` to quit.

## Troubleshooting Installation

### Common Issues

#### 1. Missing Dependencies
```bash
# If you get import errors, reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### 2. ChromaDB Setup Issues
```bash
# If manual initialization fails
python core/embeddings/semantic_embed.py
python core/embeddings/embed_tmdb_parameters.py

# Clear ChromaDB and reinitialize if corrupted
rm -rf chroma_db/
python core/embeddings/semantic_embed.py
python core/embeddings/embed_tmdb_parameters.py

# Verify collections were created
python -c "import chromadb; client = chromadb.PersistentClient(path='chroma_db'); print(client.list_collections())"
```

#### 3. API Key Problems
- **TMDB**: Ensure you're using the "Read Access Token", not the v3 API key
- **OpenAI**: Verify your account has billing enabled
- **Environment**: Check that `.env` file is in the project root

#### 4. Network/Firewall Issues
- Ensure outbound HTTPS access to:
  - `api.themoviedb.org` (TMDB API)
  - `api.openai.com` (OpenAI API)
  - `huggingface.co` (model downloads)

#### 5. Python Version Issues
```bash
# Check Python version
python --version
# Should be 3.8 or higher

# If too old, install newer Python
# On Ubuntu/Debian:
sudo apt update && sudo apt install python3.9

# On macOS:
brew install python@3.9
```

### Memory Requirements

- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM for better performance
- **Storage**: ~2GB for models and database

### Performance Optimization

#### 1. First Run Setup
The first execution downloads ML models (~500MB) and builds the vector database. Subsequent runs are much faster.

#### 2. Debug Mode
- **Production**: Keep `DEBUG_MODE = False` in `app.py` for clean output
- **Development**: Set `DEBUG_MODE = True` for detailed logging

#### 3. Model Caching
Models are cached locally after first download. If having issues:
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/sentence-transformers/
```

## System Requirements

### Operating Systems
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS (10.14+)
- Windows 10/11

### Hardware Requirements
- **CPU**: Modern multi-core processor
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 5GB free space
- **Network**: Broadband internet for API calls

## Security Considerations

### API Key Protection
- Never commit `.env` files to version control
- Use environment variables in production
- Rotate API keys regularly
- Monitor API usage for unexpected activity

### Local Data
- ChromaDB stores embeddings locally (no sensitive data)
- No movie/TV content is downloaded or stored
- Query history is not persistently stored

## Next Steps

After successful installation:

1. **Read the [User Guide](user-guide.md)** for query examples and usage patterns
2. **Explore [Configuration Options](configuration.md)** for customization
3. **Check [Troubleshooting](troubleshooting.md)** if you encounter issues

## Getting Help

If you encounter issues not covered here:

1. **Check [Troubleshooting Guide](troubleshooting.md)**
2. **Review [GitHub Issues](https://github.com/FinnMacCumail/tmdbGPT/issues)**
3. **Create a new issue** with:
   - Operating system and Python version
   - Complete error message
   - Steps to reproduce the problem

## Updating TMDBGPT

To update to the latest version:

```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

If major updates change the ChromaDB schema:
```bash
rm -rf chroma_db/
python app.py  # Rebuilds database
```