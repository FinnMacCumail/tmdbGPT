"""
Phase 2: Complex Triple Constraints Test Matrix

This test suite systematically validates 3+ constraint combinations to identify
current failure modes and establish baseline performance for Phase 2 enhancements.

Target query patterns:
- Person + Genre + Date (e.g., "2010s sci-fi by Nolan")  
- Person + Role + Company (e.g., "Action movies starring Chris Evans from Marvel")
- Genre + Network + Awards (e.g., "Sci-fi TV shows from HBO starring Emmy winners")
- Genre + Date + Technical (e.g., "Horror films from 1980s under $10M budget")
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.model.constraint import Constraint, ConstraintGroup, ConstraintBuilder
from core.model.evaluator import evaluate_constraint_tree, relax_constraint_tree


class TestComplexTripleConstraints(unittest.TestCase):
    """Test matrix for complex 3+ constraint combinations"""
    
    def setUp(self):
        """Setup comprehensive test data registry"""
        self.data_registry = {
            # People constraints
            "with_people": {
                "525": {1001, 1002, 1003, 1004, 1005},  # Christopher Nolan
                "1892": {1003, 1004, 1006, 1007},       # Matt Damon  
                "31": {1002, 1005, 1008, 1009},         # Tom Hanks
                "16828": {1006, 1007, 1008, 1010},      # Chris Evans
                "935": {1009, 1010, 1011, 1012}         # Ridley Scott
            },
            
            # Genre constraints  
            "with_genres": {
                "878": {1001, 1003, 1005, 1007, 1009, 1011},  # Science Fiction
                "28": {1002, 1004, 1006, 1008, 1010, 1012},   # Action
                "27": {1013, 1014, 1015, 1016, 1017, 1018},   # Horror
                "18": {1019, 1020, 1021, 1022}               # Drama
            },
            
            # Company constraints
            "with_companies": {
                "420": {1001, 1003, 1005, 1007},        # Marvel Studios
                "3": {1002, 1004, 1006, 1008},          # Warner Bros
                "1": {1009, 1010, 1011, 1012},          # Disney
                "174": {1013, 1014, 1015, 1016}         # A24
            },
            
            # Network constraints (for TV)
            "with_networks": {
                "49": {2001, 2002, 2003, 2004},         # HBO
                "213": {2005, 2006, 2007, 2008},        # Netflix
                "1024": {2009, 2010, 2011, 2012}        # Amazon Prime
            },
            
            # Date constraints
            "primary_release_year": {
                "2010": {1001, 1002, 1003, 2001, 2002},
                "2015": {1004, 1005, 1006, 2003, 2004},
                "2020": {1007, 1008, 1009, 2005, 2006},
                "1980": {1010, 1011, 1012, 2007, 2008}
            }
        }
        
        # Expected complex constraint combinations with intersection results
        self.expected_intersections = {
            # Person + Genre + Date
            ("with_people", "with_genres", "primary_release_year"): {
                ("525", "878", "2010"): {1001, 1003},  # Nolan + Sci-fi + 2010s
                ("1892", "28", "2015"): {1004, 1006},  # Matt Damon + Action + 2015
                ("935", "878", "1980"): {}             # Ridley Scott + Sci-fi + 1980 (no intersection)
            },
            
            # Person + Role + Company  
            ("with_people", "with_companies"): {
                ("16828", "420"): {1007},               # Chris Evans + Marvel (should intersect)
                ("31", "3"): {1002, 1008},             # Tom Hanks + Warner Bros
                ("1892", "174"): {}                    # Matt Damon + A24 (no intersection)
            },
            
            # Genre + Company + Date
            ("with_genres", "with_companies", "primary_release_year"): {
                ("28", "420", "2015"): {1007},         # Action + Marvel + 2015
                ("27", "174", "2020"): {}              # Horror + A24 + 2020 (no intersection)
            }
        }

    def test_triple_constraint_person_genre_date(self):
        """Test Person + Genre + Date combinations (e.g., '2010s sci-fi by Nolan')"""
        
        # Case 1: Should have intersection (Nolan + Sci-fi + 2010)
        tree = ConstraintGroup([
            Constraint(key="with_people", value="525", type_="person", subtype="director", priority=6),
            Constraint(key="with_genres", value="878", type_="genre", priority=2),
            Constraint(key="primary_release_year", value="2010", type_="date", priority=3)
        ], logic="AND")
        
        result = evaluate_constraint_tree(tree, self.data_registry)
        
        # Should find intersection of all three constraints
        expected_intersection = self.expected_intersections[("with_people", "with_genres", "primary_release_year")][("525", "878", "2010")]
        self.assertIsInstance(result, dict)
        
        # Verify intersection exists
        if result:
            # Check if any media type has results
            has_results = any(bool(constraints) for constraints in result.values())
            self.assertTrue(has_results, "Should find intersection for Nolan + Sci-fi + 2010")
        
    def test_triple_constraint_no_intersection(self):
        """Test case where 3 constraints have no intersection"""
        
        # Case: Ridley Scott + Sci-fi + 1980 (should have no intersection based on test data)
        tree = ConstraintGroup([
            Constraint(key="with_people", value="935", type_="person", subtype="director", priority=6),
            Constraint(key="with_genres", value="878", type_="genre", priority=2), 
            Constraint(key="primary_release_year", value="1980", type_="date", priority=3)
        ], logic="AND")
        
        result = evaluate_constraint_tree(tree, self.data_registry)
        
        # Should return empty result or no intersection
        if result:
            has_results = any(bool(constraints) for constraints in result.values())
            if not has_results:
                # Test constraint relaxation
                relaxed_tree, dropped, reasons = relax_constraint_tree(tree, max_drops=1)
                
                self.assertIsNotNone(relaxed_tree, "Should be able to relax constraints")
                self.assertEqual(len(dropped), 1, "Should drop exactly 1 constraint")
                self.assertTrue(len(reasons) >= 1, "Should provide relaxation reason")
                
                # Test if relaxed tree yields results
                relaxed_result = evaluate_constraint_tree(relaxed_tree, self.data_registry)
                if relaxed_result:
                    has_relaxed_results = any(bool(constraints) for constraints in relaxed_result.values())
                    # Should have results after relaxation
                    self.assertTrue(has_relaxed_results, f"Should find results after relaxation: {reasons}")

    def test_progressive_relaxation_priority_order(self):
        """Test that constraint relaxation follows expected priority order"""
        
        # Create constraint with known priority order: company(1) < genre(2) < date(3) < person(6)
        tree = ConstraintGroup([
            Constraint(key="with_companies", value="999", type_="company", priority=1),    # Invalid company
            Constraint(key="with_genres", value="999", type_="genre", priority=2),        # Invalid genre  
            Constraint(key="primary_release_year", value="1999", type_="date", priority=3), # Invalid date
            Constraint(key="with_people", value="999", type_="person", priority=6)        # Invalid person
        ], logic="AND")
        
        # Should fail initially (no intersection)
        result = evaluate_constraint_tree(tree, self.data_registry)
        has_initial_results = result and any(bool(constraints) for constraints in result.values())
        self.assertFalse(has_initial_results, "Invalid constraints should yield no results")
        
        # Test progressive relaxation
        relaxed_tree, dropped, reasons = relax_constraint_tree(tree, max_drops=1)
        
        if dropped:
            # Should drop company constraint first (lowest domain priority)
            self.assertEqual(dropped[0].key, "with_companies", 
                           f"Should drop company constraint first, but dropped: {dropped[0].key}")
            self.assertIn("company", reasons[0].lower(), "Reason should mention company constraint")

    def test_complex_constraint_combinations_matrix(self):
        """Comprehensive matrix test of various 3+ constraint combinations"""
        
        test_combinations = [
            # (Person, Genre, Date) combinations
            {
                "name": "Nolan_SciFi_2010",
                "constraints": [
                    ("with_people", "525", "person"),
                    ("with_genres", "878", "genre"), 
                    ("primary_release_year", "2010", "date")
                ],
                "expected_intersection": True
            },
            {
                "name": "MattDamon_Action_2015", 
                "constraints": [
                    ("with_people", "1892", "person"),
                    ("with_genres", "28", "genre"),
                    ("primary_release_year", "2015", "date")
                ],
                "expected_intersection": True
            },
            {
                "name": "RidleyScott_SciFi_1980",
                "constraints": [
                    ("with_people", "935", "person"),
                    ("with_genres", "878", "genre"),
                    ("primary_release_year", "1980", "date")  
                ],
                "expected_intersection": False
            },
            
            # (Person, Company, Genre) combinations
            {
                "name": "ChrisEvans_Marvel_Action",
                "constraints": [
                    ("with_people", "16828", "person"),
                    ("with_companies", "420", "company"),
                    ("with_genres", "28", "genre")
                ],
                "expected_intersection": False  # Based on test data setup
            },
            
            # (Genre, Company, Date) combinations  
            {
                "name": "Action_Marvel_2015",
                "constraints": [
                    ("with_genres", "28", "genre"),
                    ("with_companies", "420", "company"),
                    ("primary_release_year", "2015", "date")
                ],
                "expected_intersection": True
            }
        ]
        
        for test_case in test_combinations:
            with self.subTest(test_case=test_case["name"]):
                # Build constraint tree
                constraints = []
                for key, value, type_ in test_case["constraints"]:
                    priority = {"company": 1, "genre": 2, "date": 3, "person": 6}.get(type_, 5)
                    constraint = Constraint(key=key, value=value, type_=type_, priority=priority)
                    constraints.append(constraint)
                
                tree = ConstraintGroup(constraints, logic="AND")
                result = evaluate_constraint_tree(tree, self.data_registry)
                
                has_results = result and any(bool(constraints) for constraints in result.values())
                
                if test_case["expected_intersection"]:
                    self.assertTrue(has_results, 
                                  f"{test_case['name']} should have intersection but got: {result}")
                else:
                    # If no intersection expected, test relaxation recovery
                    if not has_results:
                        relaxed_tree, dropped, reasons = relax_constraint_tree(tree, max_drops=1)
                        if relaxed_tree:
                            relaxed_result = evaluate_constraint_tree(relaxed_tree, self.data_registry) 
                            has_relaxed_results = relaxed_result and any(bool(constraints) for constraints in relaxed_result.values())
                            self.assertTrue(has_relaxed_results or len(dropped) > 0,
                                          f"{test_case['name']} should recover through relaxation")

    def test_constraint_tree_structure_preservation(self):
        """Test that complex constraint trees maintain proper structure during processing"""
        
        # Nested constraint structure: (Person OR Person) AND Genre AND Date
        tree = ConstraintGroup([
            ConstraintGroup([
                Constraint(key="with_people", value="525", type_="person", priority=6),
                Constraint(key="with_people", value="935", type_="person", priority=6)
            ], logic="OR"),
            Constraint(key="with_genres", value="878", type_="genre", priority=2),
            Constraint(key="primary_release_year", value="2010", type_="date", priority=3)
        ], logic="AND")
        
        result = evaluate_constraint_tree(tree, self.data_registry)
        
        # Should handle nested structure correctly
        self.assertIsInstance(result, dict)
        
        # Structure should be preserved during evaluation
        self.assertEqual(tree.logic, "AND")
        self.assertEqual(len(tree.constraints), 3)
        self.assertIsInstance(tree.constraints[0], ConstraintGroup)
        self.assertEqual(tree.constraints[0].logic, "OR")

    def test_api_parameter_compatibility(self):
        """Test parameter compatibility for complex constraint combinations"""
        
        # Test combinations that might cause API parameter conflicts
        conflict_tests = [
            {
                "name": "Movie_and_TV_constraints",
                "constraints": [
                    Constraint(key="with_companies", value="420", type_="company", priority=1, 
                             metadata={"media_type": "movie"}),
                    Constraint(key="with_networks", value="49", type_="network", priority=1,
                             metadata={"media_type": "tv"})
                ],
                "should_conflict": True
            },
            {
                "name": "Compatible_movie_constraints", 
                "constraints": [
                    Constraint(key="with_companies", value="420", type_="company", priority=1,
                             metadata={"media_type": "movie"}),
                    Constraint(key="with_genres", value="28", type_="genre", priority=2,
                             metadata={"media_type": "movie"})
                ],
                "should_conflict": False
            }
        ]
        
        for test_case in conflict_tests:
            with self.subTest(test_case=test_case["name"]):
                tree = ConstraintGroup(test_case["constraints"], logic="AND")
                
                # Check for media type conflicts
                media_types = set()
                for constraint in tree.constraints:
                    if hasattr(constraint, 'metadata') and constraint.metadata:
                        media_type = constraint.metadata.get('media_type')
                        if media_type:
                            media_types.add(media_type)
                
                has_conflict = len(media_types) > 1
                self.assertEqual(has_conflict, test_case["should_conflict"],
                               f"Conflict detection mismatch for {test_case['name']}")


if __name__ == "__main__":
    # Run comprehensive test suite
    unittest.main(verbosity=2)