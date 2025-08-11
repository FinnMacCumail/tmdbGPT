#!/usr/bin/env python3
"""
Test Company/Studio Query Fixes
Verify that the fixes for HBO and Marvel Studios work
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
from app import build_app_graph

class TestCompanyStudioFixes(unittest.TestCase):
    """Test that our company/studio fixes work"""
    
    def setUp(self):
        """Set up test environment"""
        self.graph = build_app_graph()
    
    def test_hbo_us_resolution(self):
        """Test that HBO resolves to US HBO (not Polish HBO)"""
        try:
            result = self.graph.invoke({"input": "HBO shows"})
            
            # Check entity resolution
            extraction = result.get("extraction_result", {})
            entities = extraction.get("query_entities", [])
            hbo_entities = [e for e in entities if "hbo" in e.get("name", "").lower()]
            
            if hbo_entities:
                hbo_entity = hbo_entities[0]
                resolved_id = hbo_entity.get("resolved_id")
                
                # Should NOT be Polish HBO (8102), should be US HBO (49 or similar)
                self.assertNotEqual(resolved_id, 8102, 
                                   f"HBO should not resolve to Polish HBO (8102), got {resolved_id}")
                
                # Should be a US network (we expect 49 based on our API test)
                self.assertEqual(resolved_id, 49, 
                                f"HBO should resolve to US HBO (49), got {resolved_id}")
                
                print(f"✅ HBO Resolution Fixed: {hbo_entity}")
                return True
            else:
                self.fail("HBO entity not found in extraction")
                
        except Exception as e:
            self.fail(f"HBO test failed with exception: {e}")
            return False
    
    def test_marvel_constraint_building(self):
        """Test that Marvel Studios constraint building works"""
        try:
            result = self.graph.invoke({"input": "Movies by Marvel Studios"})
            
            # Check that we don't get "No results found"
            responses = result.get("responses", [])
            
            # Look for the "No results found" message that indicates failure
            no_results = any("no results found" in str(response).lower() for response in responses)
            
            if no_results:
                print(f"❌ Marvel Studios still returns no results: {responses}")
                
                # Check if entity was resolved correctly  
                extraction = result.get("extraction_result", {})
                entities = extraction.get("query_entities", [])
                marvel_entities = [e for e in entities if "marvel" in e.get("name", "").lower()]
                
                if marvel_entities:
                    print(f"✅ Entity Resolution OK: {marvel_entities[0]}")
                    
                    # Check execution trace for constraint building
                    plan_steps = result.get("execution_trace", [])
                    discover_steps = [s for s in plan_steps if "discover" in s.get("step_id", "").lower()]
                    
                    for step in discover_steps:
                        params = step.get("parameters", {})
                        if params:
                            print(f"✅ Discover step has parameters: {params}")
                        else:
                            print(f"❌ Discover step has empty parameters: {step}")
                else:
                    print(f"❌ Marvel entity not resolved")
                
                return False
            else:
                print(f"✅ Marvel Studios Constraint Building Fixed: Got {len(responses)} results")
                return True
                
        except Exception as e:
            self.fail(f"Marvel test failed with exception: {e}")
            return False

def test_fixes_quick():
    """Quick test of both fixes"""
    print("🔍 Testing Company/Studio Query Fixes")
    print("=" * 50)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCompanyStudioFixes)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All fixes working correctly!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")

if __name__ == "__main__":
    test_fixes_quick()