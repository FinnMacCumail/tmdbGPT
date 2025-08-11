#!/usr/bin/env python3
"""
Test Hulu Search Fallback Approach
Test if search-based routing works for Hulu Originals
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
from dotenv import load_dotenv

# Load environment  
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
headers = {"Authorization": f"Bearer {TMDB_API_KEY}"}

def test_direct_search():
    """Test direct TMDB search for Hulu originals"""
    print("🔍 Testing Direct TMDB Search...")
    print("=" * 50)
    
    search_terms = [
        "Hulu original series",
        "Hulu original", 
        "The Handmaid's Tale",
        "Only Murders in the Building",
        "Little Fires Everywhere"
    ]
    
    for term in search_terms:
        print(f"\nSearching for: '{term}'")
        
        response = requests.get(
            "https://api.themoviedb.org/3/search/multi",
            headers=headers,
            params={"query": term}
        )
        
        if response.status_code == 200:
            results = response.json().get("results", [])
            print(f"Found {len(results)} results:")
            
            for result in results[:5]:  # Show top 5
                media_type = result.get("media_type", "unknown")
                name = result.get("name") or result.get("title", "Unknown")
                year = ""
                if result.get("first_air_date"):
                    year = f" ({result.get('first_air_date')[:4]})"
                elif result.get("release_date"):
                    year = f" ({result.get('release_date')[:4]})"
                
                print(f"   {media_type}: {name}{year}")
        else:
            print(f"   Search failed: {response.status_code}")

def test_symbol_free_routing():
    """Test if symbol-free routing is triggered"""
    print("\n🔄 Testing Symbol-Free Routing...")
    print("=" * 50)
    
    try:
        from core.planner.plan_utils import is_symbol_free_query
        from app import AppState
        
        # Mock state for Hulu Originals
        state = AppState(
            input="Hulu Originals",
            extraction_result={
                "query_entities": [],  # Might be empty if entity resolution fails
                "entities": [],
                "question_type": "list"
            }
        )
        
        result = is_symbol_free_query(state)
        print(f"is_symbol_free_query('Hulu Originals') = {result}")
        
        if result:
            print("✅ Symbol-free routing should be triggered")
        else:
            print("❌ Symbol-free routing NOT triggered")
        
    except Exception as e:
        print(f"❌ Error testing symbol-free routing: {e}")

def test_full_query_flow():
    """Test the complete updated query flow"""
    print("\n🎬 Testing Full Query Flow...")
    print("=" * 50)
    
    try:
        from app import build_app_graph
        
        graph = build_app_graph()
        result = graph.invoke({"input": "Hulu Originals"})
        
        # Check plan steps
        plan_steps = result.get("execution_trace", [])
        print("Plan steps generated:")
        for step in plan_steps:
            step_id = step.get("step_id", "Unknown")
            endpoint = step.get("endpoint", "Unknown") 
            params = step.get("parameters", {})
            print(f"   {step_id}: {endpoint}")
            if params:
                print(f"      Params: {params}")
        
        # Check if streaming search was used
        streaming_steps = [s for s in plan_steps if "streaming_search" in s.get("step_id", "")]
        if streaming_steps:
            print("\n✅ Streaming search fallback was used!")
            for step in streaming_steps:
                print(f"   {step}")
        else:
            print("\n❌ Streaming search fallback was NOT used")
        
        # Check responses
        responses = result.get("responses", [])
        print(f"\n📊 Final responses: {len(responses)} items")
        for i, response in enumerate(responses[:5]):
            if isinstance(response, dict):
                title = response.get("title") or response.get("name", "Unknown")
                print(f"   {i+1}. {title}")
            else:
                print(f"   {i+1}. {str(response)[:50]}...")
        
    except Exception as e:
        print(f"❌ Query flow test failed: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing Hulu Search Fallback Implementation")
    print("=" * 60)
    
    test_direct_search()
    test_symbol_free_routing() 
    test_full_query_flow()
    
    print(f"\n🎯 Expected Outcome:")
    print("- Symbol-free routing detects 'Hulu Originals' pattern")
    print("- Search fallback uses /search/multi with 'Hulu original series'")  
    print("- Results show actual Hulu original shows, not Japanese content")

if __name__ == "__main__":
    main()