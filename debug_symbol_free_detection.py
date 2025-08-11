#!/usr/bin/env python3
"""
Debug Symbol-Free Detection Logic
Check why is_symbol_free_query returns False for BBC/Legendary
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.planner.plan_utils import is_symbol_free_query
from app import AppState

def debug_symbol_free_detection(query_text, extraction_result):
    """Debug symbol-free detection logic"""
    
    print(f"\n🔍 Debug Symbol-Free Detection: '{query_text}'")
    print("=" * 50)
    
    # Create a mock state
    state = AppState(
        input=query_text,
        extraction_result=extraction_result,
        intended_media_type="both"
    )
    
    # Extract components
    query_entities = extraction_result.get("query_entities", [])
    entities = extraction_result.get("entities", [])
    question_type = extraction_result.get("question_type", "")
    
    print(f"📋 Extraction Components:")
    print(f"   query_entities: {query_entities}")
    print(f"   entities: {entities}")  
    print(f"   question_type: {question_type}")
    
    # Check symbol-free conditions
    print(f"\n🧮 Symbol-Free Logic Check:")
    
    # Condition 1: Single-entity fact queries
    is_single_fact = question_type == "fact" and len(query_entities) == 1
    print(f"   Single-entity fact query: {is_single_fact}")
    
    # Condition 2: Single company/network queries  
    is_single_company_network = False
    if len(query_entities) == 1:
        entity_type = query_entities[0].get("type")
        is_single_company_network = entity_type in {"company", "network"}
    print(f"   Single company/network query: {is_single_company_network}")
    print(f"      Query entities count: {len(query_entities)}")
    if query_entities:
        print(f"      Entity type: {query_entities[0].get('type')}")
    
    # Condition 3: No symbolic entities
    non_symbolic_entities = {"movie", "tv"}
    real_entities = [e for e in entities if e not in non_symbolic_entities]
    no_symbolic_entities = not query_entities and not real_entities
    print(f"   No symbolic entities: {no_symbolic_entities}")
    print(f"      Real entities: {real_entities}")
    
    # Final result
    result = is_symbol_free_query(state)
    print(f"\n🎯 Final Result: is_symbol_free_query() = {result}")
    
    return result

def test_queries():
    """Test problematic queries"""
    
    # BBC Shows extraction result (from debug output)
    bbc_extraction = {
        "intents": ["discovery.filtered"],
        "entities": ["tv", "network"],
        "query_entities": [
            {
                "name": "BBC",
                "type": "network", 
                "resolved_id": 4796,
                "resolved_type": "network"
            }
        ],
        "question_type": "list",
        "response_format": "ranked_list",
        "media_type": "both"
    }
    
    # Legendary extraction result (from debug output)
    legendary_extraction = {
        "intents": ["discovery.filtered"],
        "entities": ["movie", "company"],
        "query_entities": [
            {
                "name": "Legendary Entertainment",
                "type": "company",
                "resolved_id": 185226, 
                "resolved_type": "company"
            }
        ],
        "question_type": "list", 
        "response_format": "ranked_list",
        "media_type": "movie"
    }
    
    debug_symbol_free_detection("BBC shows", bbc_extraction)
    debug_symbol_free_detection("Movies by Legendary Entertainment", legendary_extraction)

if __name__ == "__main__":
    test_queries()