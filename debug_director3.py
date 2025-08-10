#!/usr/bin/env python3

# Debug the constraint tree building

import sys
import os
sys.path.append(os.getcwd())

from core.llm.extractor import extract_entities_and_intents
from core.constraint_model import ConstraintBuilder

def debug_constraint_building():
    query = "Who directed Inception?"
    
    # Extract entities like the app does
    extraction = extract_entities_and_intents(query)
    print(f"Extraction result: {extraction}")
    
    # Check constraint tree building logic
    query_entities = extraction.get("query_entities", [])
    has_person_entities = any(e.get("type") == "person" for e in query_entities)
    
    print(f"Query entities: {query_entities}")
    print(f"Has person entities: {has_person_entities}")
    print(f"Question type: {extraction.get('question_type')}")
    
    # Check if constraint tree should be built
    should_build_constraints = extraction.get("question_type") != "fact" or has_person_entities
    print(f"Should build constraints: {should_build_constraints}")
    
    if should_build_constraints:
        builder = ConstraintBuilder()
        constraint_tree = builder.build_from_query_entities(query_entities)
        print(f"Constraint tree: {constraint_tree}")
        print(f"Constraint tree logic: {constraint_tree.logic}")
        print(f"Constraints: {[str(c) for c in constraint_tree.constraints]}")
    else:
        print("Empty constraint tree would be used")

if __name__ == "__main__":
    debug_constraint_building()