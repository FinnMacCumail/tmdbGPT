#!/usr/bin/env python3
"""
Test Symbol-Free Company/Network Routing
Quick test to verify the fixes work
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import build_app_graph

def test_hbo_shows():
    """Test HBO shows query with symbol-free routing"""
    print("🔍 Testing 'HBO shows'...")
    
    try:
        graph = build_app_graph()
        result = graph.invoke({"input": "HBO shows"})
        
        # Check responses
        responses = result.get("responses", [])
        
        if responses and len(responses) > 0 and not any("no results" in str(r).lower() for r in responses):
            print(f"✅ HBO Shows SUCCESS: Got {len(responses)} results")
            
            # Show first few results
            for i, response in enumerate(responses[:3]):
                if isinstance(response, dict):
                    title = response.get("title", "Unknown")
                    print(f"   {i+1}. {title}")
                else:
                    print(f"   {i+1}. {str(response)[:100]}...")
            return True
        else:
            print(f"❌ HBO Shows FAILED: {responses}")
            return False
            
    except Exception as e:
        print(f"❌ HBO Shows ERROR: {e}")
        return False

def test_marvel_movies():
    """Test Marvel Studios query with symbol-free routing"""  
    print("\n🔍 Testing 'Movies by Marvel Studios'...")
    
    try:
        graph = build_app_graph()
        result = graph.invoke({"input": "Movies by Marvel Studios"})
        
        # Check responses  
        responses = result.get("responses", [])
        
        if responses and len(responses) > 0 and not any("no results" in str(r).lower() for r in responses):
            print(f"✅ Marvel Movies SUCCESS: Got {len(responses)} results")
            
            # Show first few results
            for i, response in enumerate(responses[:3]):
                if isinstance(response, dict):
                    title = response.get("title", "Unknown")
                    print(f"   {i+1}. {title}")
                else:
                    print(f"   {i+1}. {str(response)[:100]}...")
            return True
        else:
            print(f"❌ Marvel Movies FAILED: {responses}")
            return False
            
    except Exception as e:
        print(f"❌ Marvel Movies ERROR: {e}")
        return False

def main():
    print("🎬 Testing Symbol-Free Company/Network Routing")
    print("=" * 60)
    
    hbo_success = test_hbo_shows()
    marvel_success = test_marvel_movies()
    
    print(f"\n📊 Results:")
    print(f"HBO Shows: {'✅' if hbo_success else '❌'}")
    print(f"Marvel Movies: {'✅' if marvel_success else '❌'}")
    
    if hbo_success and marvel_success:
        print("\n🎉 All tests passed! Company/Network symbol-free routing is working!")
    else:
        print("\n⚠️ Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main()