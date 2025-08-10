#!/usr/bin/env python3

# Debug the full flow to understand what's happening

import sys
import os
sys.path.append(os.getcwd())

from core.execution_state import AppState
from core.llm.extractor import extract_entities_and_intents
from core.constraint_model import ConstraintBuilder
from core.model.constraint import ConstraintGroup

def debug_full_flow():
    query = "Who directed Inception?"
    print(f"=== DEBUGGING: {query} ===\n")
    
    # Step 1: Entity extraction
    extraction = extract_entities_and_intents(query)
    print(f"1. ENTITY EXTRACTION:")
    print(f"   Question type: {extraction.get('question_type')}")
    print(f"   Entities: {extraction.get('entities', [])}")
    print(f"   Query entities: {extraction.get('query_entities', [])}")
    print()
    
    # Step 2: Check constraint tree logic
    query_entities = extraction.get("query_entities", [])
    entities = extraction.get("entities", [])
    has_person_entities = any(e.get("type") == "person" for e in query_entities) or "person" in entities
    
    print(f"2. CONSTRAINT TREE LOGIC:")
    print(f"   Has person entities: {has_person_entities}")
    print(f"   Should build constraints: {extraction.get('question_type') != 'fact' or has_person_entities}")
    print()
    
    # Step 3: Build constraint tree
    if extraction.get("question_type") != "fact" or has_person_entities:
        builder = ConstraintBuilder()
        constraint_tree = builder.build_from_query_entities(query_entities)
        print(f"3. CONSTRAINT TREE:")
        print(f"   Logic: {constraint_tree.logic}")
        print(f"   Constraints: {[str(c) for c in constraint_tree.constraints]}")
    else:
        constraint_tree = ConstraintGroup([], logic="AND")
        print(f"3. CONSTRAINT TREE:")
        print(f"   Using empty constraint tree")
    print()
    
    # Step 4: Check what path this would take
    if constraint_tree.constraints:
        print(f"4. EXPECTED PATH: Discovery path (constraint-based)")
    else:
        print(f"4. EXPECTED PATH: Simple details path")
    print()

if __name__ == "__main__":
    debug_full_flow()