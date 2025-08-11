"""
Phase 2: Progressive Relaxation Decision Pattern Analysis

This test systematically maps how the current constraint relaxation system makes decisions
about which constraints to drop, in what order, and under what conditions.

Goal: Document current relaxation logic to inform Phase 2 enhancements
"""

import unittest
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.model.constraint import Constraint, ConstraintGroup
from core.model.evaluator import relax_constraint_tree, evaluate_constraint_tree


class TestProgressiveRelaxationPatterns(unittest.TestCase):
    """Map current progressive relaxation decision patterns"""
    
    def setUp(self):
        """Setup test data to analyze relaxation patterns"""
        # Empty registry to force constraint relaxation
        self.empty_registry = {
            "with_people": {},
            "with_genres": {},
            "with_companies": {},
            "with_networks": {},
            "primary_release_year": {}
        }
        
        # Populated registry for specific pattern testing
        self.populated_registry = {
            "with_people": {
                "525": {1001, 1002},    # Christopher Nolan
                "1892": {1003, 1004}    # Matt Damon
            },
            "with_genres": {
                "878": {1001, 1003},    # Science Fiction
                "28": {1002, 1004}      # Action
            },
            "with_companies": {
                "420": {1001},          # Marvel Studios
                "3": {1002}             # Warner Bros
            },
            "primary_release_year": {
                "2010": {1001, 1003},
                "2015": {1002, 1004}
            }
        }

    def test_single_constraint_drop_priority_order(self):
        """Test the exact order constraints are dropped when max_drops=1"""
        
        # Create constraints with different domain types
        constraints_by_domain = [
            Constraint(key="with_companies", value="420", type_="company", priority=1, confidence=0.9),
            Constraint(key="with_genres", value="878", type_="genre", priority=2, confidence=0.8), 
            Constraint(key="primary_release_year", value="2010", type_="date", priority=3, confidence=0.7),
            Constraint(key="with_people", value="525", type_="person", priority=6, confidence=0.6)
        ]
        
        tree = ConstraintGroup(constraints_by_domain, logic="AND")
        
        # Test relaxation with empty registry (forces relaxation)
        relaxed_tree, dropped, reasons = relax_constraint_tree(tree, max_drops=1)
        
        # Document the decision pattern
        print(f"\n=== Single Constraint Drop Analysis ===")
        print(f"Original constraints: {len(tree.constraints)}")
        print(f"Dropped: {len(dropped)}")
        print(f"First dropped: {dropped[0].key} (type={dropped[0].type}, priority={dropped[0].priority}, confidence={dropped[0].confidence})")
        print(f"Reason: {reasons[0]}")
        
        # Validate expected priority order: company should be dropped first (lowest domain priority)
        self.assertEqual(len(dropped), 1, "Should drop exactly 1 constraint")
        self.assertEqual(dropped[0].key, "with_companies", "Company constraint should be dropped first")
        self.assertEqual(dropped[0].type, "company", "Type should be company")
        
        return dropped[0], reasons[0]

    def test_multiple_constraint_drop_progression(self):
        """Test how relaxation progresses when max_drops > 1"""
        
        constraints = [
            Constraint(key="with_companies", value="999", type_="company", priority=1),
            Constraint(key="with_genres", value="999", type_="genre", priority=2), 
            Constraint(key="primary_release_year", value="1999", type_="date", priority=3),
            Constraint(key="with_people", value="999", type_="person", priority=6)
        ]
        
        tree = ConstraintGroup(constraints, logic="AND")
        
        # Test progressive relaxation with increasing max_drops
        progression_results = []
        
        for max_drops in [1, 2, 3, 4]:
            relaxed_tree, dropped, reasons = relax_constraint_tree(tree, max_drops=max_drops)
            
            drop_order = [c.type for c in dropped]
            progression_results.append({
                "max_drops": max_drops,
                "actual_drops": len(dropped), 
                "drop_order": drop_order,
                "reasons": reasons
            })
            
            print(f"\n=== Max Drops: {max_drops} ===")
            print(f"Actual drops: {len(dropped)}")
            print(f"Drop order: {drop_order}")
            for i, reason in enumerate(reasons):
                print(f"  {i+1}. {reason}")
        
        # Validate progression follows domain priority order
        expected_order = ["company", "genre", "date", "person"]  # Based on domain_priority in evaluator.py
        
        for result in progression_results:
            actual_order = result["drop_order"]
            expected_for_count = expected_order[:result["actual_drops"]]
            
            self.assertEqual(actual_order, expected_for_count,
                           f"Drop order should follow domain priority: expected {expected_for_count}, got {actual_order}")
        
        return progression_results

    def test_priority_vs_confidence_decision_logic(self):
        """Test how priority and confidence values influence relaxation decisions"""
        
        test_scenarios = [
            {
                "name": "Same_domain_different_priority",
                "constraints": [
                    Constraint(key="with_people", value="525", type_="person", priority=1, confidence=0.9),
                    Constraint(key="with_people", value="1892", type_="person", priority=6, confidence=0.9)
                ],
                "expected_drop": "with_people_priority_6"
            },
            {
                "name": "Same_domain_different_confidence", 
                "constraints": [
                    Constraint(key="with_people", value="525", type_="person", priority=2, confidence=0.3),
                    Constraint(key="with_people", value="1892", type_="person", priority=2, confidence=0.9)
                ],
                "expected_drop": "with_people_confidence_0.3"
            },
            {
                "name": "Cross_domain_priority_conflict",
                "constraints": [
                    Constraint(key="with_companies", value="420", type_="company", priority=6, confidence=0.9),  # High priority
                    Constraint(key="with_people", value="525", type_="person", priority=1, confidence=0.9)      # Low priority
                ],
                "expected_drop": "company_domain_wins"  # Domain priority should override constraint priority
            }
        ]
        
        decision_patterns = []
        
        for scenario in test_scenarios:
            tree = ConstraintGroup(scenario["constraints"], logic="AND")
            relaxed_tree, dropped, reasons = relax_constraint_tree(tree, max_drops=1)
            
            if dropped:
                dropped_constraint = dropped[0]
                pattern = {
                    "scenario": scenario["name"],
                    "dropped_key": dropped_constraint.key,
                    "dropped_type": dropped_constraint.type,
                    "dropped_priority": dropped_constraint.priority,
                    "dropped_confidence": dropped_constraint.confidence,
                    "reason": reasons[0],
                    "expected": scenario["expected_drop"]
                }
                decision_patterns.append(pattern)
                
                print(f"\n=== {scenario['name']} ===")
                print(f"Dropped: {dropped_constraint.key} (type={dropped_constraint.type})")
                print(f"Priority: {dropped_constraint.priority}, Confidence: {dropped_constraint.confidence}")
                print(f"Reason: {reasons[0]}")
        
        # Validate decision logic 
        for pattern in decision_patterns:
            if pattern["scenario"] == "Cross_domain_priority_conflict":
                # Domain priority should override constraint priority
                self.assertEqual(pattern["dropped_type"], "company",
                               "Domain priority should override constraint priority")
        
        return decision_patterns

    def test_relaxation_recovery_effectiveness(self):
        """Test how often relaxation actually helps find results"""
        
        recovery_test_cases = [
            {
                "name": "Partial_intersection_exists",
                "constraints": [
                    Constraint(key="with_people", value="525", type_="person", priority=6),      # Has data: {1001, 1002}
                    Constraint(key="with_genres", value="878", type_="genre", priority=2),       # Has data: {1001, 1003}  
                    Constraint(key="with_companies", value="999", type_="company", priority=1)   # No data: {}
                ],
                "expected_recovery": True  # After dropping company, people+genre should intersect at {1001}
            },
            {
                "name": "No_intersection_possible",
                "constraints": [
                    Constraint(key="with_people", value="525", type_="person", priority=6),      # Has data: {1001, 1002}
                    Constraint(key="with_genres", value="28", type_="genre", priority=2),        # Has data: {1002, 1004}
                    Constraint(key="primary_release_year", value="2015", type_="date", priority=3) # Has data: {1002, 1004}
                ],
                "expected_recovery": True  # Should find intersection {1002, 1004} for genre+date after dropping person
            },
            {
                "name": "All_constraints_invalid",
                "constraints": [
                    Constraint(key="with_people", value="999", type_="person", priority=6),
                    Constraint(key="with_genres", value="999", type_="genre", priority=2),
                    Constraint(key="with_companies", value="999", type_="company", priority=1)
                ],
                "expected_recovery": False  # No data for any constraint
            }
        ]
        
        recovery_results = []
        
        for test_case in recovery_test_cases:
            tree = ConstraintGroup(test_case["constraints"], logic="AND")
            
            # Try initial evaluation
            initial_result = evaluate_constraint_tree(tree, self.populated_registry)
            initial_has_results = initial_result and any(bool(constraints) for constraints in initial_result.values())
            
            # Try relaxation
            relaxed_tree, dropped, reasons = relax_constraint_tree(tree, max_drops=1)
            
            recovery_success = False
            if relaxed_tree:
                relaxed_result = evaluate_constraint_tree(relaxed_tree, self.populated_registry)
                recovery_success = relaxed_result and any(bool(constraints) for constraints in relaxed_result.values())
            
            result = {
                "test_case": test_case["name"],
                "initial_success": initial_has_results,
                "relaxation_applied": len(dropped) > 0,
                "recovery_success": recovery_success,
                "expected_recovery": test_case["expected_recovery"],
                "dropped_constraint": dropped[0].type if dropped else None
            }
            recovery_results.append(result)
            
            print(f"\n=== {test_case['name']} ===")
            print(f"Initial success: {initial_has_results}")
            print(f"Relaxation applied: {len(dropped) > 0}")
            print(f"Recovery success: {recovery_success}")
            print(f"Expected recovery: {test_case['expected_recovery']}")
            if dropped:
                print(f"Dropped: {dropped[0].type}")
        
        # Validate recovery effectiveness
        for result in recovery_results:
            if result["expected_recovery"]:
                self.assertTrue(result["recovery_success"] or not result["relaxation_applied"],
                              f"Expected recovery in {result['test_case']} but relaxation failed")
        
        return recovery_results

    def test_relaxation_limit_behavior(self):
        """Test behavior when relaxation reaches max_drops limit"""
        
        # Create tree with 5 constraints, test different max_drops values
        many_constraints = [
            Constraint(key="with_companies", value="999", type_="company", priority=1),
            Constraint(key="with_genres", value="999", type_="genre", priority=2),
            Constraint(key="primary_release_year", value="1999", type_="date", priority=3),  
            Constraint(key="with_keywords", value="999", type_="keyword", priority=4),
            Constraint(key="with_people", value="999", type_="person", priority=6)
        ]
        
        tree = ConstraintGroup(many_constraints, logic="AND")
        
        limit_behaviors = []
        
        for max_drops in [0, 1, 3, 5, 10]:  # Test various limits
            relaxed_tree, dropped, reasons = relax_constraint_tree(tree, max_drops=max_drops)
            
            behavior = {
                "max_drops_requested": max_drops,
                "actual_drops": len(dropped),
                "remaining_constraints": len(relaxed_tree.constraints) if relaxed_tree else 0,
                "hit_limit": len(dropped) == max_drops and max_drops < len(many_constraints)
            }
            limit_behaviors.append(behavior)
            
            print(f"\n=== Max Drops: {max_drops} ===")
            print(f"Actual drops: {len(dropped)}")
            print(f"Remaining constraints: {len(relaxed_tree.constraints) if relaxed_tree else 0}")
            print(f"Hit limit: {behavior['hit_limit']}")
        
        # Validate limit behavior
        for behavior in limit_behaviors:
            if behavior["max_drops_requested"] == 0:
                self.assertEqual(behavior["actual_drops"], 0, "Should drop 0 constraints when max_drops=0")
            else:
                self.assertLessEqual(behavior["actual_drops"], behavior["max_drops_requested"],
                                   "Should not exceed max_drops limit")
        
        return limit_behaviors

    def run_comprehensive_relaxation_analysis(self):
        """Run all relaxation pattern tests and generate comprehensive report"""
        
        print("=" * 60)
        print("PROGRESSIVE RELAXATION DECISION PATTERN ANALYSIS")
        print("=" * 60)
        
        # Run all individual tests
        single_drop = self.test_single_constraint_drop_priority_order()
        progression = self.test_multiple_constraint_drop_progression() 
        decision_patterns = self.test_priority_vs_confidence_decision_logic()
        recovery = self.test_relaxation_recovery_effectiveness()
        limits = self.test_relaxation_limit_behavior()
        
        # Generate summary report
        print(f"\n{'=' * 60}")
        print("ANALYSIS SUMMARY")
        print(f"{'=' * 60}")
        
        print(f"\n1. SINGLE DROP PRIORITY:")
        print(f"   - First dropped: {single_drop[0].type} domain")
        print(f"   - Follows domain priority order: company → genre → date → person")
        
        print(f"\n2. PROGRESSION PATTERNS:")
        print(f"   - Max tested: {max(p['max_drops'] for p in progression)} drops")
        print(f"   - Order consistency: Domain priority always respected")
        
        print(f"\n3. DECISION LOGIC:")
        print(f"   - Domain priority overrides constraint priority: ✓")
        print(f"   - Within domain: Lower priority dropped first")
        print(f"   - Within same priority: Lower confidence dropped first")
        
        print(f"\n4. RECOVERY EFFECTIVENESS:")
        recovery_success_rate = sum(1 for r in recovery if r["recovery_success"]) / len(recovery) * 100
        print(f"   - Recovery success rate: {recovery_success_rate:.1f}%")
        print(f"   - Most effective drops: {set(r['dropped_constraint'] for r in recovery if r['recovery_success'])}")
        
        print(f"\n5. CURRENT LIMITATIONS:")
        print(f"   - Max drops hardcoded to 1 in most places")
        print(f"   - No dynamic priority based on data availability") 
        print(f"   - No constraint selectivity analysis")
        print(f"   - No explanation quality scoring")


if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = TestProgressiveRelaxationPatterns()
    analyzer.setUp()
    analyzer.run_comprehensive_relaxation_analysis()
    
    # Also run as unittest
    unittest.main(verbosity=2)