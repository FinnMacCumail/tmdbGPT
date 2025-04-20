import os
from datetime import datetime
from app import build_app_graph
from response_formatter import ResponseFormatter

# List of test queries
queries = [
    "Tell me about Sofia Coppola",
    "How many movies has Tom Hanks been in?",
    "Show me films directed by Christopher Nolan",
    "Find action movies from 2023",
    "What are some popular sci-fi films?",
    "Best romance movies with Anne Hathaway",
    "Movies rated above 8 with Brad Pitt and Edward Norton",
    "Popular Netflix shows",
    "Find TV series with Bryan Cranston",
    "Shows airing on HBO",
    "What’s trending this week?",
    "Popular movies today",
    "Top-rated films on TMDB right now",
    "Search for The Godfather",
    "Get details about Interstellar",
    "Show me the trailer for Avatar: The Way of Water",
    "Find movies starring DiCaprio directed by Scorsese",
    "Which movies feature both Matt Damon and Ben Affleck?",
    "List Netflix movies with Ryan Reynolds",
    "Comedy films with time travel",
    "Horror movies rated above 7.5",
    "Sci-fi movies released between 2010 and 2020",
    "Films like Hereditary and Midsommar",
    "What are some good movies?"
]

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Timestamped log file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"logs/batch_test_{timestamp}.log"

def log_line(text, file):
    print(text)
    file.write(text + "\n")

if __name__ == "__main__":
    print("🧪 Starting TMDB-GPT Batch Test...")
    graph = build_app_graph()

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"🧪 TMDB-GPT Batch Test Log — {timestamp}\n\n")

        for i, query in enumerate(queries, 1):
            log_line(f"\n🔹 Test {i}: {query}", log_file)
            try:
                result = graph.invoke({"input": query})
                responses = result.get("responses", [])
                if responses:
                    formatted = ResponseFormatter.format_responses(responses)
                    log_line("✅ Response:", log_file)
                    for line in formatted:
                        log_line(f"  {line}", log_file)
                else:
                    log_line("⚠️ No response returned.", log_file)
            except Exception as e:
                log_line(f"❌ Error on query: {query}", log_file)
                log_line(f"   → {e}", log_file)

    print(f"📝 Results saved to: {log_path}")