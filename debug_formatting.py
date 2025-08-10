#!/usr/bin/env python3

# Debug the formatting logic specifically

import sys
import os
sys.path.append(os.getcwd())

def debug_formatting():
    # Simulate the data that would be passed to formatting
    query_text = "who directed inception?"
    
    # Test the keyword detection
    is_director_question = any(keyword in query_text for keyword in ["direct", "director", "directed"])
    print(f"Query: {query_text}")
    print(f"Is director question: {is_director_question}")
    
    # Simulate a movie response with directors
    movie_response = {
        "type": "movie_summary",
        "title": "Inception",
        "directors": ["Christopher Nolan"],
        "release_date": "2010-07-15"
    }
    
    print(f"Movie response: {movie_response}")
    print(f"Has directors: {bool(movie_response.get('directors'))}")
    
    # Test the formatting logic
    if is_director_question:
        if movie_response.get("directors"):
            directors = ", ".join(movie_response["directors"])
            result = f"🎬 {movie_response['title']} was directed by {directors}."
            print(f"Expected result: {result}")
        else:
            print("No directors found in response")
    else:
        print("Not detected as director question")

if __name__ == "__main__":
    debug_formatting()