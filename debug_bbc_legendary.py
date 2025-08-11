#!/usr/bin/env python3
"""
Debug BBC/Legendary Response Processing
Check why these queries return entity profiles instead of content
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import build_app_graph

def debug_query_flow(query):
    """Debug the complete flow for a query"""
    print(f"\n🔍 Debugging: '{query}'")
    print("=" * 50)
    
    try:
        graph = build_app_graph()
        result = graph.invoke({"input": query})
        
        # Check extraction
        extraction = result.get("extraction_result", {})
        query_entities = extraction.get("query_entities", [])
        
        print(f"📋 Query Entities:")
        for entity in query_entities:
            name = entity.get("name")
            entity_type = entity.get("type")
            resolved_id = entity.get("resolved_id")
            print(f"   {name} → {entity_type} (ID: {resolved_id})")
        
        # Check plan steps
        plan_steps = result.get("execution_trace", [])
        print(f"\n🛠️ Plan Steps:")
        for step in plan_steps:
            step_id = step.get("step_id", "Unknown")
            endpoint = step.get("endpoint", "Unknown")
            params = step.get("parameters", {})
            status = step.get("status", "Unknown")
            
            print(f"   {step_id}: {endpoint}")
            if params:
                print(f"      Params: {params}")
            print(f"      Status: {status}")
        
        # Check if symbol-free routing was used
        symbol_free_steps = [s for s in plan_steps if any(keyword in s.get("step_id", "") 
                            for keyword in ["company_movies", "network_shows", "company_details", "network_details"])]
        
        if symbol_free_steps:
            print(f"\n✅ Symbol-free routing detected:")
            for step in symbol_free_steps:
                print(f"   {step.get('step_id')}: {step.get('endpoint')}")
                print(f"      Params: {step.get('parameters', {})}")
        else:
            print(f"\n❌ No symbol-free routing detected")
        
        # Check responses
        responses = result.get("responses", [])
        print(f"\n📊 Final Responses ({len(responses)}):")
        for i, response in enumerate(responses[:3]):
            if isinstance(response, dict):
                response_type = response.get("type", "unknown")
                title = response.get("title", "Unknown")
                source = response.get("source", "Unknown")
                print(f"   {i+1}. {title} (type: {response_type}, source: {source})")
            else:
                print(f"   {i+1}. {str(response)[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🔍 Debugging BBC/Legendary Response Processing")
    
    # Test the problematic queries
    debug_query_flow("BBC shows")
    debug_query_flow("Movies by Legendary Entertainment")
    
    print(f"\n📝 Analysis:")
    print("- Check if symbol-free routing is being triggered")
    print("- Check if discover endpoints are called with correct parameters") 
    print("- Check if responses are movie/TV content or entity profiles")

if __name__ == "__main__":
    main()