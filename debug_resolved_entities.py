#!/usr/bin/env python3
"""
Debug Resolved Entities Structure
Check what keys are created in resolved_entities dict
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from app import build_app_graph

def debug_resolved_entities():
    """Debug the resolved entities structure"""
    
    print("🔍 Debugging Resolved Entities Structure")
    print("=" * 60)
    
    graph = build_app_graph()
    
    # Test Marvel Studios query
    result = graph.invoke({"input": "Movies by Marvel Studios"})
    
    # Check what we have in different result components
    print("📋 Extraction Result (query_entities):")
    extraction = result.get("extraction_result", {})
    query_entities = extraction.get("query_entities", [])
    for entity in query_entities:
        print(f"  {entity}")
    
    print(f"\n📦 Resolved Entities Dict:")
    # This should be state.resolved_entities that gets passed to dependency manager
    # But it's not directly in the final result, so let's check intermediate state
    
    # Let's look for any clues about resolved_entities in the execution trace
    plan_steps = result.get("execution_trace", [])
    
    # Look for debug output that might show resolved_entities
    for step in plan_steps:
        if "resolved" in str(step).lower():
            print(f"  Step with 'resolved': {step.get('step_id', 'Unknown')}")
    
    print(f"\n🔄 State Flow Analysis:")
    print("Expected flow:")
    print("1. query_entities with resolved_id → app.py:resolve_entities() → resolved_entities dict")
    print("2. resolved_entities dict → dependency_manager.expand_plan_with_dependencies()")
    print("3. dependency_manager looks for 'company_id' key but app.py creates what key?")
    
    # Let's manually trace the app.py logic
    print(f"\n🧮 Manual Key Creation Logic:")
    for entity in query_entities:
        if "resolved_id" in entity:
            resolved_type = entity.get("resolved_type", entity["type"])
            resolved_id = entity["resolved_id"]
            key = f"{resolved_type}_id"
            print(f"  Entity: {entity['name']} → Key: '{key}' → Value: {resolved_id}")

if __name__ == "__main__":
    debug_resolved_entities()