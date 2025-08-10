#!/usr/bin/env python3

# Debug the LLM call specifically

import sys
import os
sys.path.append(os.getcwd())

from core.llm.extractor import extract_entities_and_intents

def debug_llm_call():
    query = "Who directed Inception?"
    print(f"Testing query: {query}")
    
    try:
        result = extract_entities_and_intents(query)
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_llm_call()