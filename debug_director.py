#!/usr/bin/env python3

# Debug script to understand why director queries aren't working

import sys
import os
sys.path.append(os.getcwd())

from core.execution_state import AppState
from app import build_app_graph

def debug_director_query():
    print("=== DEBUGGING DIRECTOR QUERY ===")
    
    # Initialize application
    graph = build_app_graph()
    
    query = "Who directed Inception?"
    print(f"Testing query: {query}")
    
    # Process query
    result = graph.invoke({"input": query})
    
    print(f"\nFinal result: {result}")
    
    return result

if __name__ == "__main__":
    debug_director_query()