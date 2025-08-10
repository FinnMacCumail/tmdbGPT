# User Guide

This comprehensive guide will help you get the most out of TMDBGPT, from basic queries to advanced features.

## Getting Started

### Launching TMDBGPT

After completing the [installation](installation.md), start the application:

```bash
cd tmdbGPT
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

You'll see the prompt:
```
Ask something (or type 'exit' to quit):
```

### Your First Query

Try asking a simple question:
```
Tell me about the movie The Dark Knight
```

TMDBGPT will show friendly progress indicators and then provide detailed information about the movie.

## Understanding Query Types

TMDBGPT can handle various types of questions about movies and TV shows:

### 1. Information Queries

Ask for details about specific movies, shows, or people:

```
Tell me about Inception
Who is Christopher Nolan?
What is Breaking Bad about?
```

**Example Response:**
```
🎬 Inception (2010)
   Director: Christopher Nolan
   Starring: Leonardo DiCaprio, Marion Cotillard, Tom Hardy
   Overview: Dom Cobb is a skilled thief who steals corporate secrets...
   Rating: 8.8/10
```

### 2. Search Queries

Find movies or shows matching specific criteria:

```
Movies directed by Steven Spielberg
Movies starring Tom Hanks
Leonardo DiCaprio movies
```

### 3. Multi-Entity Queries

Combine multiple people, genres, or constraints:

```
Movies starring Leonardo DiCaprio directed by Martin Scorsese
Movies starring Brad Pitt produced by Plan B Entertainment
Movies with Al Pacino, Robert De Niro, Val Kilmer
```

### 4. Count Queries 

Ask about quantities and statistics:

#### Person Role Counts
```
How many movies has Quentin Tarantino directed?
How many films has Sofia Coppola directed?
How many movies has Leonardo DiCaprio starred in?
```

#### TV Show Attributes 
```
How many seasons does Breaking Bad have?
How many episodes does The Office have?
How many seasons does Game of Thrones have?
How many episodes does Friends have?
```

### 5. Fact Queries ✅ GREATLY ENHANCED

Get specific factual information with high accuracy:

#### Movie Facts
```
# Directors and Crew
Who directed The Godfather?
Who wrote Inception?
Who composed The Dark Knight?

# Technical Details  
How long is Titanic?              # Runtime in minutes
What genre is The Matrix?         # Action, Science Fiction
What was Avengers budget?         # Budget amount
What year was Blade Runner released? # Release year
```

#### TV Show Facts 
```
# Creators and Cast
Who created Breaking Bad?
Who starred in The Office?
Who wrote Friends?

# Show Attributes
How many seasons does Breaking Bad have?      # Season count
How many episodes does Game of Thrones have?  # Episode count
When did The Office first air?               # First air date
```

### 6. Studio and Production Queries

Find content from specific studios or by specific roles:

```
Movies scored by Hans Zimmer
Films with music by John Williams
Movies produced by Jerry Bruckheimer
Films from Studio Ghibli
```

## Query Patterns and Examples

### Basic Movie Queries

```
# Movie information
"Tell me about Pulp Fiction"
"Give me details about The Matrix"

# Director searches  
"Movies directed by Christopher Nolan"
"Christopher Nolan's filmography"

# Actor searches
"Movies starring Tom Hanks"
"Leonardo DiCaprio movies"
```

### TV Show Queries

```
# Show information
"Tell me about Game of Thrones"
"What is The Office about?"

# Network searches
"Best HBO series"
"Netflix original shows"

# Genre searches  
"Crime drama TV shows"
"Comedy series with high ratings"
```

### Complex Multi-Constraint Queries

```
# Genre + Director
"Thrillers directed by David Fincher"

# Composer + Studio
"Movies scored by Hans Zimmer made by Warner Bros"

# Writer + Composer
"Movies written by Quentin Tarantino with music by Ennio Morricone"

# Genre + Network + Actor
"Sci-fi shows on Netflix starring Millie Bobby Brown"
```

### Role-Specific Queries

TMDBGPT now has **comprehensive role support** for both movies and TV shows, with 90-95% success rates:

#### TV Show Role Queries 

```
# TV Creators and Cast
"Who created Breaking Bad?"
"Who starred in The Office?"
"Who starred in Breaking Bad?"
"Who starred in Friends?"

# TV Writers and Producers  
"Who wrote Breaking Bad?"
"Who produced Game of Thrones?"
"Who produced The Office?"
```

#### Movie Role Queries 

```
# Directors and Writers
"Who directed The Dark Knight?"
"Who wrote Inception?"
"Who wrote Pulp Fiction?"

# Composers and Producers
"Who composed The Dark Knight?"
"Who produced Pulp Fiction?"
"Who produced The Godfather?"

# Cast Members
"Who starred in Inception?"
"Who starred in The Godfather?"
```

#### Legacy Production Queries

```
# Studio and Network Queries
"Movies scored by Hans Zimmer"
"Films with music by John Williams"
"TV shows directed by David Lynch aired on Showtime"
"Movies produced by Jerry Bruckheimer"
"Films from Studio Ghibli"
```

## Understanding Results

### User-Friendly Mode (Default)

When `DEBUG_MODE = False` in `app.py`, you'll see:

1. **Progress Indicators**: Friendly messages showing processing stages
2. **Clean Results**: Well-formatted movie/TV information
3. **Simple Output**: Easy-to-read results without technical details
4. **Enhanced Role Data**: Complete cast, crew, and creator information ✨

Example (Movie Discovery):
```
🔍 Understanding your question...
🎭 Identifying people, movies, and details...
🔎 Looking up information...
📚 Gathering context...
🗓️ Planning search strategy...
🎬 Searching movies and shows...
✨ Preparing your results...
📋 Formatting your results...

============================================================
🎬 The Departed (2006)
   Director: Martin Scorsese
   Starring: Leonardo DiCaprio, Matt Damon, Jack Nicholson
   Overview: An undercover cop and a spy in the police attempt...
   Rating: 8.5/10

🎬 Shutter Island (2010)
   Director: Martin Scorsese  
   Starring: Leonardo DiCaprio, Mark Ruffalo
   Overview: In 1954, a U.S. Marshal investigates the disappearance...
   Rating: 8.2/10
============================================================
```

Example (Role Query) ✨:
```
🔍 Understanding your question...
🎭 Identifying people, movies, and details...
🔎 Looking up information...
📚 Gathering context...
🗓️ Planning search strategy...
🎬 Searching movies and shows...
✨ Preparing your results...
📋 Formatting your results...

============================================================
📺 Breaking Bad was created by Vince Gilligan, Michelle MacLaren, Mark Johnson.
============================================================
```

### Debug Mode (For Developers)

When `DEBUG_MODE = True` in `app.py`, you'll also see:

1. **🧠 DEBUGGING SUMMARY REPORT**: Complete technical analysis
2. **Execution Trace**: Step-by-step processing details  
3. **Constraint Analysis**: How queries were interpreted
4. **Fallback Information**: What relaxations were applied

## Advanced Features

### Query Refinement

If TMDBGPT can't find exact matches, it will automatically:

1. **Relax Constraints**: Drop less important filters (company before genre before people)
2. **Semantic Fallback**: Find related content when exact matches fail
3. **Provide Alternatives**: Suggest similar results

Example:
```
Query: "Netflix thriller movies from 1995 directed by David Fincher"
Result: No exact matches found, showing David Fincher thrillers from similar periods
```

### Multi-Step Processing

For complex queries, TMDBGPT:

1. **Resolves Entities**: Identifies people, movies, companies
2. **Plans Execution**: Determines required API calls
3. **Validates Results**: Confirms all constraints are met
4. **Intersects Data**: Combines information from multiple sources

### Role Validation

TMDBGPT verifies that people actually have the roles mentioned:

```
Query: "Movies directed by Leonardo DiCaprio"
Process: 
- Finds Leonardo DiCaprio's credits
- Filters only directing roles
- Returns verified director credits
```

## Tips for Better Results

### Be Specific but Natural

✅ **Good**: "Movies starring Leonardo DiCaprio directed by Martin Scorsese"
❌ **Less Good**: "Scorsese DiCaprio films"

### Use Natural Language

✅ **Good**: "Best horror movies from 2023"  
✅ **Also Good**: "Top-rated horror films released in 2023"
❌ **Less Good**: "horror 2023 movies rating>7"

### Multiple People Queries

✅ **Good**: "Movies with Al Pacino, Robert De Niro, Val Kilmer"
✅ **Good**: "Movies starring Brad Pitt produced by Plan B Entertainment"

### Time Periods

✅ **Good**: "Movies from the 2010s"
✅ **Good**: "Films released between 2010 and 2019"
✅ **Good**: "Recent movies from 2023"

### Genres and Types

✅ **Good**: "Sci-fi thriller movies"
✅ **Good**: "Crime drama TV shows"  
✅ **Good**: "Animated family films"

## Common Query Patterns

### Discovery Queries

```
"Thrillers directed by David Fincher"
"Movies scored by Hans Zimmer"
```

### Multi-Role Queries

```
"Movies written by Quentin Tarantino with music by Ennio Morricone"
"Movies scored by Hans Zimmer made by Warner Bros"  
"TV shows directed by David Lynch aired on Showtime"
```

### Actor-Focused Queries

```
"Movies starring Tom Hanks"
"Leonardo DiCaprio movies"
"Movies with Al Pacino, Robert De Niro, Val Kilmer"
```

### Director-Production Queries

```
"Movies starring Leonardo DiCaprio directed by Martin Scorsese"
"Movies starring Brad Pitt produced by Plan B Entertainment"
"Christopher Nolan's filmography"
```

## Troubleshooting Queries

### If You Get No Results

1. **Check Spelling**: Verify names are spelled correctly
2. **Be More General**: Try removing some constraints
3. **Use Alternative Names**: Try different forms of names
4. **Check Time Periods**: Verify release years

### If Results Seem Wrong

1. **Check Debug Mode**: Enable debugging to see processing details
2. **Refine Query**: Be more specific about what you want
3. **Use Exact Titles**: Include full movie/show titles in quotes

### If Processing Seems Slow

1. **Complex Queries Take Time**: Multi-constraint queries require more processing
2. **First Run Slower**: Initial setup and model loading takes time
3. **Network Issues**: Check internet connection for API calls

## Exiting the Application

To stop TMDBGPT, type any of these:
```
exit
quit
```

Or press `Ctrl+C` to force quit.

## Next Steps

- Explore [Configuration Options](configuration.md) for customization
- Learn about [Advanced Usage](advanced-usage.md) for power users
- Check [Troubleshooting Guide](troubleshooting.md) for common issues
- Read [Architecture Documentation](architecture.md) to understand how it works

---

**Happy querying!** 🎬✨ TMDBGPT is ready to help you discover movies and TV shows in the most natural way possible.