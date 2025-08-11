#!/usr/bin/env python3
"""
Debug Marvel Studios Constraint Flow
Investigates why Marvel Studios constraint building/validation fails in tmdbGPT
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from app import build_app_graph, DEBUG_MODE

# Enable debug mode for detailed output
import app
app.DEBUG_MODE = True

def debug_marvel_constraint_flow():
    """Debug the full Marvel Studios query flow"""
    
    print("🔍 Debugging Marvel Studios Query Flow")
    print("=" * 60)
    
    graph = build_app_graph()
    
    try:
        result = graph.invoke({"input": "Movies by Marvel Studios"})
        
        # Extract and analyze key components
        extraction = result.get("extraction_result", {})
        plan_steps = result.get("execution_trace", [])
        responses = result.get("responses", [])
        
        print("📋 Extraction Result:")
        print(json.dumps(extraction, indent=2))
        
        print("\n🛠️ Plan Steps Analysis:")
        for i, step in enumerate(plan_steps):
            step_id = step.get("step_id", "Unknown")
            endpoint = step.get("endpoint", "Unknown")
            status = step.get("status", "Unknown")
            parameters = step.get("parameters", {})
            
            print(f"  {i+1}. {step_id}")
            print(f"     Endpoint: {endpoint}")
            print(f"     Status: {status}")
            if parameters:
                print(f"     Parameters: {parameters}")
            print()
        
        print("📊 Final Responses:")
        for i, response in enumerate(responses):
            print(f"  {i+1}. {response}")
        
        # Analyze specific failure patterns
        print("\n🔍 Failure Analysis:")
        
        # Check if entity was resolved
        entities = extraction.get("query_entities", [])
        marvel_entities = [e for e in entities if "marvel" in e.get("name", "").lower()]
        
        if marvel_entities:
            marvel_entity = marvel_entities[0]
            print(f"✅ Marvel entity resolved: {marvel_entity}")
            
            if "resolved_id" in marvel_entity:
                resolved_id = marvel_entity["resolved_id"]
                print(f"✅ Marvel resolved to ID: {resolved_id}")
                
                # Check if discover step was created with correct parameters
                discover_steps = [s for s in plan_steps if "discover" in s.get("step_id", "").lower()]
                
                if discover_steps:
                    print("✅ Discover steps found:")
                    for step in discover_steps:
                        params = step.get("parameters", {})
                        endpoint = step.get("endpoint", "")
                        print(f"     {endpoint} with params: {params}")
                        
                        # Check for correct parameter usage
                        if "with_companies" in params:
                            companies = params["with_companies"]
                            if str(resolved_id) in str(companies):
                                print(f"✅ Correct company ID {resolved_id} found in with_companies parameter")
                            else:
                                print(f"❌ Marvel ID {resolved_id} NOT found in with_companies: {companies}")
                        else:
                            print("❌ No with_companies parameter found")
                else:
                    print("❌ No discover steps found - constraint building failed")
            else:
                print("❌ Marvel entity not resolved to ID")
        else:
            print("❌ Marvel entity not extracted")
        
        # Check why fallback was used
        fallback_steps = [s for s in plan_steps if "fallback" in s.get("step_id", "").lower()]
        if fallback_steps:
            print(f"\n⚠️ Fallback used ({len(fallback_steps)} steps):")
            for step in fallback_steps:
                print(f"     {step.get('step_id')} - {step.get('endpoint')}")
                print(f"     Parameters: {step.get('parameters', {})}")
    
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_marvel_constraint_flow()