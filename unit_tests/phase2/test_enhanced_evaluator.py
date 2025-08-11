"""
Test the enhanced constraint evaluator against the failing Phase 2 cases.

This validates that the progressive filtering approach fixes the multi-constraint
intersection failures identified in the original test matrix.
"""

import unittest
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.model.constraint import Constraint, ConstraintGroup
from core.model.evaluator_enhanced import (
    evaluate_constraint_tree_enhanced,
    relax_constraint_tree_enhanced
)


class TestEnhancedConstraintEvaluator(unittest.TestCase):
    """Test enhanced evaluator against known failure cases"""
    
    def setUp(self):
        """Setup test data that mimics the failing Action_Marvel_2015 case"""
        self.data_registry = {
            # Genre constraint: Action movies
            "with_genres": {
                "28": {1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008}  # Action genre
            },
            
            # Company constraint: Marvel Studios  
            "with_companies": {
                "420": {1001, 1003, 1005, 1007, 1009, 1011}  # Marvel movies
            },
            
            # Date constraint: 2015 releases
            "primary_release_year": {
                "2015": {1002, 1004, 1006, 1007, 1008, 1010}  # 2015 releases
            }
        }
        
        # Expected intersection for Action(28) + Marvel(420) + 2015:
        # Action: {1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008}
        # Marvel: {1001, 1003, 1005, 1007, 1009, 1011} 
        # 2015:   {1002, 1004, 1006, 1007, 1008, 1010}
        # Intersection: {1007} (the only movie that appears in all three sets)

    def test_enhanced_evaluator_fixes_triple_constraint_failure(self):
        """Test that enhanced evaluator fixes the Action_Marvel_2015 failure case"""
        
        # Create the failing constraint combination
        constraints = [
            Constraint(key="with_genres", value="28", type_="genre", priority=2),
            Constraint(key="with_companies", value="420", type_="company", priority=1), 
            Constraint(key="primary_release_year", value="2015", type_="date", priority=3)
        ]
        
        tree = ConstraintGroup(constraints, logic="AND")
        
        # Test with enhanced evaluator
        result = evaluate_constraint_tree_enhanced(tree, self.data_registry, debug=True)
        
        print(f"\nEnhanced evaluator result: {result}")
        
        # Should find the intersection {1007}
        has_results = result and any(bool(constraints) for constraints in result.values())
        self.assertTrue(has_results, f"Enhanced evaluator should find intersection but got: {result}")
        
        # Verify specific intersection
        movie_results = result.get("movie", {})
        if movie_results:
            # Should have results for all constraint types
            genre_results = movie_results.get("with_genres", set())
            company_results = movie_results.get("with_companies", set())  
            date_results = movie_results.get("primary_release_year", set())
            
            print(f"Genre results: {genre_results}")
            print(f"Company results: {company_results}")
            print(f"Date results: {date_results}")
            
            # All should contain the intersection ID(s)
            expected_intersection = {1007}
            self.assertEqual(genre_results, expected_intersection)
            self.assertEqual(company_results, expected_intersection)
            self.assertEqual(date_results, expected_intersection)

    def test_enhanced_evaluator_vs_original_comparison(self):
        """Compare enhanced vs original evaluator on the same failing case"""
        
        from core.model.evaluator import evaluate_constraint_tree as original_evaluate
        
        constraints = [
            Constraint(key="with_genres", value="28", type_="genre", priority=2,
                     metadata={"media_type": "movie"}),
            Constraint(key="with_companies", value="420", type_="company", priority=1,
                     metadata={"media_type": "movie"}),
            Constraint(key="primary_release_year", value="2015", type_="date", priority=3,
                     metadata={"media_type": "movie"})
        ]
        
        tree = ConstraintGroup(constraints, logic="AND")
        
        # Test original evaluator
        original_result = original_evaluate(tree, self.data_registry)
        
        # Test enhanced evaluator  
        enhanced_result = evaluate_constraint_tree_enhanced(tree, self.data_registry, debug=True)
        
        print(f"\nOriginal evaluator result: {original_result}")
        print(f"Enhanced evaluator result: {enhanced_result}")
        
        # Original should fail (empty results)
        original_has_results = original_result and any(bool(constraints) for constraints in original_result.values())
        self.assertFalse(original_has_results, "Original evaluator should fail on this case")
        
        # Enhanced should succeed
        enhanced_has_results = enhanced_result and any(bool(constraints) for constraints in enhanced_result.values())
        self.assertTrue(enhanced_has_results, "Enhanced evaluator should succeed on this case")

    def test_enhanced_relaxation_multiple_drops(self):
        """Test enhanced relaxation with multiple constraint drops"""
        
        # Create case where 2 constraints need to be dropped
        impossible_constraints = [
            Constraint(key="with_genres", value="999", type_="genre", priority=2),      # No data
            Constraint(key="with_companies", value="999", type_="company", priority=1), # No data  
            Constraint(key="primary_release_year", value="2015", type_="date", priority=3), # Has data
            Constraint(key="with_people", value="525", type_="person", priority=6)      # Not in registry
        ]
        
        tree = ConstraintGroup(impossible_constraints, logic="AND")
        
        # Test enhanced relaxation with multiple drops
        relaxed_tree, dropped, reasons = relax_constraint_tree_enhanced(
            tree, max_drops=3, data_registry=self.data_registry, debug=True
        )
        
        print(f"\nRelaxation results:")
        print(f"Dropped: {len(dropped)} constraints")
        for i, reason in enumerate(reasons):
            print(f"  {i+1}. {reason}")
        
        # Should drop multiple constraints
        self.assertGreater(len(dropped), 1, "Should drop multiple constraints")
        self.assertLessEqual(len(dropped), 3, "Should respect max_drops limit")
        
        # Test if relaxed tree yields results
        if relaxed_tree:
            relaxed_result = evaluate_constraint_tree_enhanced(relaxed_tree, self.data_registry)
            relaxed_has_results = relaxed_result and any(bool(constraints) for constraints in relaxed_result.values())
            
            print(f"Relaxed tree result: {relaxed_result}")
            print(f"Has results after relaxation: {relaxed_has_results}")

    def test_constraint_selectivity_analysis(self):
        """Test that constraint selectivity influences relaxation decisions"""
        
        # Create constraints with different selectivity levels
        constraints = [
            Constraint(key="with_genres", value="28", type_="genre", priority=2),          # High selectivity (8 results)
            Constraint(key="with_companies", value="420", type_="company", priority=1),    # Medium selectivity (6 results)  
            Constraint(key="primary_release_year", value="2015", type_="date", priority=3) # Medium selectivity (6 results)
        ]
        
        tree = ConstraintGroup(constraints, logic="AND")
        
        # Test that selectivity info is used in relaxation
        relaxed_tree, dropped, reasons = relax_constraint_tree_enhanced(
            tree, max_drops=1, data_registry=self.data_registry, debug=True
        )
        
        print(f"\nSelectivity-based relaxation:")
        print(f"Dropped constraint: {dropped[0].key} (type={dropped[0].type})")
        print(f"Reason: {reasons[0]}")
        
        # Should still follow domain priority (company first) but with selectivity info
        self.assertEqual(dropped[0].key, "with_companies", 
                        "Should drop company constraint first based on domain priority")
        
        # Reason should mention selectivity
        self.assertIn("selectivity", reasons[0], "Relaxation reason should include selectivity information")

    def test_progressive_filtering_step_by_step(self):
        """Test progressive filtering logic step by step"""
        
        constraints = [
            Constraint(key="with_genres", value="28", type_="genre", priority=2,
                     metadata={"media_type": "movie"}),
            Constraint(key="with_companies", value="420", type_="company", priority=1,
                     metadata={"media_type": "movie"}),
            Constraint(key="primary_release_year", value="2015", type_="date", priority=3,
                     metadata={"media_type": "movie"})
        ]
        
        tree = ConstraintGroup(constraints, logic="AND")
        
        print(f"\n=== Step-by-step progressive filtering ===")
        print(f"Genre (28) IDs: {self.data_registry['with_genres']['28']}")
        print(f"Company (420) IDs: {self.data_registry['with_companies']['420']}")
        print(f"Date (2015) IDs: {self.data_registry['primary_release_year']['2015']}")
        
        # Calculate expected intersection manually
        genre_ids = self.data_registry['with_genres']['28']
        company_ids = self.data_registry['with_companies']['420'] 
        date_ids = self.data_registry['primary_release_year']['2015']
        
        manual_intersection = genre_ids & company_ids & date_ids
        print(f"Manual intersection: {manual_intersection}")
        
        # Test enhanced evaluator
        result = evaluate_constraint_tree_enhanced(tree, self.data_registry, debug=True)
        
        # Extract actual intersection from result
        movie_results = result.get("movie", {})
        if movie_results:
            # All constraint keys should have the same intersection
            all_result_ids = set()
            for constraint_key, id_set in movie_results.items():
                all_result_ids.update(id_set)
            
            print(f"Enhanced evaluator intersection: {all_result_ids}")
            self.assertEqual(all_result_ids, manual_intersection, 
                           "Enhanced evaluator should match manual intersection calculation")


if __name__ == "__main__":
    unittest.main(verbosity=2)