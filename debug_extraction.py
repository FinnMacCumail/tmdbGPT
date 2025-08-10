#!/usr/bin/env python3

# Debug the movie details extraction

import sys
import os
sys.path.append(os.getcwd())

from nlp.nlp_retriever import ResultExtractor

def debug_extraction():
    # Simulate the JSON response that would come from TMDB API
    # This is based on the credits data from the debug output
    json_data = {
        "id": 27205,
        "title": "Inception",
        "overview": "A thief who steals corporate secrets...",
        "release_date": "2010-07-15",
        "vote_average": 8.369,
        "runtime": 148,
        "genres": [{"id": 28, "name": "Action"}],
        "budget": 160000000,
        "credits": {
            "crew": [
                {
                    "id": 525,
                    "name": "Christopher Nolan", 
                    "job": "Director",
                    "department": "Directing"
                },
                {
                    "id": 556,
                    "name": "Emma Thomas", 
                    "job": "Producer",
                    "department": "Production"
                }
            ]
        }
    }
    
    endpoint = "/movie/27205?append_to_response=credits"
    
    print("Testing movie details extraction...")
    print(f"Input JSON has credits: {'credits' in json_data}")
    print(f"Crew members: {len(json_data['credits']['crew'])}")
    
    # Test the extraction
    result = ResultExtractor._extract_movie_details(json_data, endpoint)
    
    print(f"Extraction result: {result}")
    
    if result and len(result) > 0:
        movie = result[0]
        print(f"Extracted directors: {movie.get('directors', 'NOT FOUND')}")
        print(f"Extracted title: {movie.get('title')}")
        print(f"Extracted runtime: {movie.get('runtime')}")
    else:
        print("No results from extraction")

if __name__ == "__main__":
    debug_extraction()