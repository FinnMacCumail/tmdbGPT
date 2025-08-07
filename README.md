# TMDBGPT 🎬

An intelligent movie and TV query system that understands natural language questions and provides accurate, detailed answers using The Movie Database (TMDB) API.

## What is TMDBGPT?

TMDBGPT is a sophisticated query planner that combines semantic search with symbolic constraints to answer complex questions about movies and TV shows. It uses AI-powered natural language understanding to interpret your questions and provides precise, validated results.

### Key Features

- **Natural Language Understanding**: Ask questions in plain English like "Movies starring Leonardo DiCaprio directed by Martin Scorsese"
- **Multi-Entity Constraint Solving**: Handle complex queries involving multiple people, genres, companies, and filters simultaneously
- **Semantic Search**: Uses ChromaDB and sentence transformers to find relevant information even with unconventional phrasing
- **Role-Aware Validation**: Accurately validates that directors, actors, writers, and other roles are correctly matched
- **Fallback & Relaxation**: Gracefully handles difficult queries by progressively relaxing constraints while maintaining relevance
- **User-Friendly Interface**: Toggle between detailed debugging mode and clean user experience
- **Comprehensive Logging**: Full traceability of query processing for debugging and transparency

## Quick Start

### Prerequisites

- Python 3.8+
- TMDB API Key ([get one here](https://www.themoviedb.org/settings/api))
- OpenAI API Key ([get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/FinnMacCumail/tmdbGPT.git
   cd tmdbGPT
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Configure environment**
   Create a `.env` file in the project root:
   ```env
   TMDB_API_KEY=your_tmdb_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

## Example Queries

TMDBGPT can handle a wide variety of natural language questions:

**Simple Queries:**
- "Tell me about the movie Inception"
- "Movies directed by Christopher Nolan"
- "Movies starring Tom Hanks"

**Complex Multi-Entity Queries:**
- "Movies starring Leonardo DiCaprio directed by Martin Scorsese"
- "Movies starring Brad Pitt produced by Plan B Entertainment"
- "Movies with Al Pacino, Robert De Niro, Val Kilmer"

**Count and Fact Queries:**
- "How many movies has Quentin Tarantino directed?"
- "Who directed The Dark Knight?"
- "How many films has Sofia Coppola directed?"

**Production and Role Queries:**
- "Movies scored by Hans Zimmer made by Warner Bros"
- "TV shows directed by David Lynch aired on Showtime"  
- "Movies written by Quentin Tarantino with music by Ennio Morricone"

## How It Works

TMDBGPT processes queries through several sophisticated stages:

1. **Query Understanding**: Uses LLM and SpaCy to extract entities (people, movies, genres) and intent
2. **Semantic Search**: Finds relevant TMDB API endpoints using ChromaDB vector search
3. **Constraint Planning**: Builds a symbolic constraint tree representing all query filters
4. **Multi-Step Execution**: Executes API calls with role-aware validation and result intersection
5. **Fallback & Relaxation**: Progressively relaxes constraints if no exact matches found
6. **Result Validation**: Verifies all results meet the original query requirements

## User Interface Modes

TMDBGPT supports two modes controlled by the `DEBUG_MODE` setting in `app.py`:

### User Mode (DEBUG_MODE = False)
Clean, friendly progress indicators:
```
🔍 Understanding your question...
🎭 Identifying people, movies, and details...
🔎 Looking up information...
📚 Gathering context...
🗓️ Planning search strategy...
🎬 Searching movies and shows...
✨ Preparing your results...
📋 Formatting your results...
```

### Debug Mode (DEBUG_MODE = True)
Comprehensive technical logging including:
- 🧠 DEBUGGING SUMMARY REPORT with full execution traces
- Constraint tree analysis
- Entity resolution details
- Step-by-step processing logs
- Fallback and relaxation information

## Project Structure

```
tmdbGPT/
├── app.py                 # Main application entry point
├── core/                  # Core system components
│   ├── embeddings/        # Semantic search and vector operations
│   ├── entity/            # Entity resolution and symbolic filtering
│   ├── execution/         # Query execution engine
│   ├── formatting/        # Response formatting and templates
│   ├── llm/               # Language model interactions
│   ├── model/             # Constraint and evaluation models
│   ├── planner/           # Query planning and constraint management
│   └── validation/        # Result validation and role checking
├── data/                  # Configuration data and examples
├── docs/                  # Documentation
├── nlp/                   # Natural language processing
├── response/              # Response handling and logging
├── unit_tests/            # Comprehensive test suite
└── utils/                 # Utility functions and tools
```

## Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[User Guide](docs/user-guide.md)** - Complete usage guide with examples
- **[Architecture](docs/architecture.md)** - System architecture and design
- **[Configuration](docs/configuration.md)** - Configuration options and tuning
- **[API Reference](docs/api-reference.md)** - Developer API documentation
- **[Contributing](docs/contributing.md)** - How to contribute to the project
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)** - Development practices and branching

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md) for details on:
- Development workflow
- Code style guidelines
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **The Movie Database (TMDB)** for providing comprehensive movie and TV data
- **OpenAI** for language model capabilities
- **ChromaDB** for vector search functionality
- **Sentence Transformers** for semantic embeddings
- **SpaCy** for natural language processing

## Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting Guide](docs/troubleshooting.md)
2. Review existing [GitHub Issues](https://github.com/FinnMacCumail/tmdbGPT/issues)
3. Create a new issue with detailed information about your problem

---

**TMDBGPT** - Making movie and TV discovery as natural as conversation! 🍿✨