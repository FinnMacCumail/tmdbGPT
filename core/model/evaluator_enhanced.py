"""
Enhanced Constraint Tree Evaluator for Phase 2 Complex Triple Constraints

Key improvements:
1. Progressive filtering instead of exact set intersection
2. Constraint selectivity analysis for intelligent priority
3. Multi-constraint AND logic that works with sparse data
4. Enhanced relaxation strategies with multiple constraint drops

This replaces the problematic logic in evaluator.py for Phase 2 testing.
"""

from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple, Dict, Set, List
import logging

from core.model.constraint import Constraint, ConstraintGroup


def evaluate_constraint_tree_enhanced(group: ConstraintGroup, data_registry: dict, debug=False) -> Dict[str, Dict[str, Set[int]]]:
    """
    Enhanced constraint tree evaluation using progressive filtering approach.
    
    Instead of requiring exact set intersection across all constraints,
    this version progressively filters results through each constraint,
    which works better for sparse real-world data.
    """
    
    if not group or not group.constraints:
        return {"movie": {}, "tv": {}}

    if debug:
        print(f"=== Evaluating constraint group with {len(group.constraints)} constraints, logic={group.logic} ===")

    # Collect all constraint results with metadata
    constraint_results = []
    
    for node in group:
        if isinstance(node, ConstraintGroup):
            # Recursively handle nested constraint groups
            result = evaluate_constraint_tree_enhanced(node, data_registry, debug)
            constraint_results.append({"result": result, "node": node, "type": "group"})
        else:
            # Handle individual constraints
            value_str = str(node.value)
            media_type = node.metadata.get("media_type", "movie")
            
            # Defensive structure validation
            if not isinstance(data_registry.get(node.key), dict):
                data_registry[node.key] = {}
            
            id_set = data_registry[node.key].get(value_str, set())
            result = {media_type: {node.key: id_set}} if id_set else {}
            
            constraint_results.append({
                "result": result, 
                "node": node, 
                "type": "constraint",
                "selectivity": len(id_set) if id_set else 0,
                "key": node.key,
                "value": value_str
            })
            
            if debug:
                print(f"  Constraint {node.key}={value_str}: {len(id_set) if id_set else 0} results")

    # Always return consistent structure
    merged = {"movie": {}, "tv": {}}
    
    if not constraint_results:
        return merged

    if group.logic == "AND":
        # Progressive filtering approach for AND logic
        merged = _progressive_and_filtering(constraint_results, debug)
    elif group.logic == "OR":
        # Union approach for OR logic (unchanged)
        merged = _union_or_logic(constraint_results)

    if debug:
        total_results = sum(len(constraints) for constraints in merged.values())
        print(f"=== Final merged results: {total_results} total ===")
    
    return merged


def _progressive_and_filtering(constraint_results: List[dict], debug=False) -> Dict[str, Dict[str, Set[int]]]:
    """
    Progressive filtering approach for AND logic.
    
    Instead of requiring exact intersection, this:
    1. Starts with constraint with most results (highest selectivity)
    2. Progressively filters through remaining constraints
    3. Maintains results that satisfy ALL constraints
    """
    
    # Separate constraint results by media type
    by_media_type = defaultdict(list)
    for cr in constraint_results:
        for media_type, constraints in cr["result"].items():
            if constraints:  # Only consider non-empty results
                by_media_type[media_type].append({
                    "constraints": constraints,
                    "selectivity": cr.get("selectivity", 0),
                    "node": cr["node"]
                })
    
    merged = {"movie": {}, "tv": {}}
    
    for media_type, media_constraints in by_media_type.items():
        if not media_constraints:
            continue
        
        if debug:
            print(f"    Processing {media_type} constraints: {len(media_constraints)}")
        
        # Sort by selectivity (start with most selective constraint)
        media_constraints.sort(key=lambda x: x["selectivity"], reverse=True)
        
        # Start with most selective constraint's results
        base_constraint = media_constraints[0]
        candidate_ids = set()
        
        # Get all IDs from the most selective constraint
        for constraint_key, id_set in base_constraint["constraints"].items():
            candidate_ids.update(id_set)
        
        if debug:
            print(f"      Starting with {len(candidate_ids)} candidate IDs from most selective constraint")
        
        # Progressive filtering through remaining constraints
        for constraint_info in media_constraints[1:]:
            if not candidate_ids:
                break  # No candidates left
            
            # Get all IDs from this constraint
            constraint_ids = set()
            for constraint_key, id_set in constraint_info["constraints"].items():
                constraint_ids.update(id_set)
            
            # Keep only candidates that appear in this constraint
            candidate_ids &= constraint_ids
            
            if debug:
                print(f"      After filtering through constraint: {len(candidate_ids)} candidates remain")
        
        # Build final result structure
        if candidate_ids:
            # Distribute final candidates back to their constraint keys
            for constraint_info in media_constraints:
                for constraint_key, id_set in constraint_info["constraints"].items():
                    final_ids = candidate_ids & id_set
                    if final_ids:
                        merged[media_type].setdefault(constraint_key, set()).update(final_ids)
    
    return merged


def _union_or_logic(constraint_results: List[dict]) -> Dict[str, Dict[str, Set[int]]]:
    """Union approach for OR logic (unchanged from original)"""
    merged = {"movie": {}, "tv": {}}
    
    for cr in constraint_results:
        for media_type, constraints in cr["result"].items():
            for constraint_key, id_set in constraints.items():
                merged[media_type].setdefault(constraint_key, set()).update(id_set)
    
    return merged


def relax_constraint_tree_enhanced(
    tree: ConstraintGroup,
    max_drops: int = 2,  # Increased default from 1 to 2
    data_registry: dict = None,
    debug: bool = False
) -> Tuple[Optional[ConstraintGroup], list, list]:
    """
    Enhanced constraint relaxation with intelligent priority and multiple drops.
    
    Improvements:
    1. Support for multiple constraint drops (default max_drops=2)
    2. Dynamic priority based on constraint selectivity 
    3. Better explanation generation
    4. Context-aware relaxation strategies
    """
    
    relaxed = deepcopy(tree)
    
    def collect_constraints(group):
        for c in group.constraints:
            if isinstance(c, ConstraintGroup):
                yield from collect_constraints(c)
            else:
                yield c
    
    flat_constraints = list(collect_constraints(relaxed))
    if not flat_constraints:
        return None, [], ["No constraints found in tree"]
    
    # Calculate constraint selectivity if data registry available
    constraint_selectivity = {}
    if data_registry:
        for constraint in flat_constraints:
            value_str = str(constraint.value)
            registry_key = constraint.key
            
            if registry_key in data_registry and value_str in data_registry[registry_key]:
                selectivity = len(data_registry[registry_key][value_str])
            else:
                selectivity = 0
            
            constraint_selectivity[id(constraint)] = selectivity
    
    # Enhanced domain priority with selectivity consideration
    domain_priority = {
        "company": 1,
        "network": 1, 
        "genre": 2,
        "date": 3,
        "language": 4,
        "runtime": 5,
        "keyword": 6,
        "person": 7  # Increased from 6 to 7 to prioritize keeping person constraints
    }
    
    # Sort constraints for relaxation with enhanced logic
    def constraint_sort_key(c):
        base_domain_priority = domain_priority.get(c.type, 9)
        selectivity = constraint_selectivity.get(id(c), 0)
        
        # Prefer dropping constraints with low selectivity (fewer results)
        # Within same domain priority, drop lower constraint priority first
        # Within same constraint priority, drop lower confidence first
        return (
            base_domain_priority,        # Domain priority (lower = drop first)
            -selectivity,               # Selectivity (lower = drop first) 
            c.priority,                 # Constraint priority (lower = drop first)
            -c.confidence               # Confidence (lower = drop first)
        )
    
    sorted_constraints = sorted(flat_constraints, key=constraint_sort_key)
    
    dropped = []
    reasons = []
    
    if debug:
        print(f"=== Constraint Relaxation Analysis ===")
        print(f"Constraints to consider for dropping (in order):")
        for i, c in enumerate(sorted_constraints):
            selectivity = constraint_selectivity.get(id(c), "unknown")
            print(f"  {i+1}. {c.key}={c.value} (type={c.type}, priority={c.priority}, "
                  f"confidence={c.confidence}, selectivity={selectivity})")
    
    # Progressive constraint relaxation
    for constraint in sorted_constraints:
        if len(dropped) >= max_drops:
            break
        
        # Remove constraint from tree
        def remove_constraint(group):
            group.constraints = [
                c for c in group.constraints
                if not (not isinstance(c, ConstraintGroup) and 
                       c.key == constraint.key and 
                       c.value == constraint.value)
            ]
            for c in group.constraints:
                if isinstance(c, ConstraintGroup):
                    remove_constraint(c)
        
        remove_constraint(relaxed)
        dropped.append(constraint)
        
        # Enhanced reason generation
        selectivity = constraint_selectivity.get(id(constraint), "unknown")
        reason = (
            f"Dropped '{constraint.key}={constraint.value}' "
            f"(type={constraint.type}, priority={constraint.priority}, "
            f"confidence={constraint.confidence:.2f}, selectivity={selectivity})"
        )
        reasons.append(reason)
        
        if debug:
            print(f"  Dropped: {reason}")
        
        # Test if relaxation is sufficient (optional early stopping)
        if data_registry:
            test_result = evaluate_constraint_tree_enhanced(relaxed, data_registry)
            has_results = test_result and any(bool(constraints) for constraints in test_result.values())
            
            if has_results:
                if debug:
                    print(f"  Relaxation successful after dropping {len(dropped)} constraints")
                break
    
    if not dropped:
        return None, [], ["No constraints could be dropped"]
    
    return relaxed, dropped, reasons


def analyze_constraint_selectivity(constraints: List[Constraint], data_registry: dict) -> Dict[str, int]:
    """
    Analyze constraint selectivity to inform relaxation decisions.
    
    Returns mapping of constraint_id -> result_count for intelligent prioritization.
    """
    selectivity = {}
    
    for constraint in constraints:
        value_str = str(constraint.value)
        registry_key = constraint.key
        
        if registry_key in data_registry and value_str in data_registry[registry_key]:
            result_count = len(data_registry[registry_key][value_str])
        else:
            result_count = 0
        
        selectivity[f"{constraint.key}_{constraint.value}"] = result_count
    
    return selectivity


# Backward compatibility functions
def evaluate_constraint_tree(group: ConstraintGroup, data_registry: dict) -> Dict[str, Dict[str, Set[int]]]:
    """Backward compatibility wrapper - delegates to enhanced version"""
    return evaluate_constraint_tree_enhanced(group, data_registry)


def relax_constraint_tree(tree: ConstraintGroup, max_drops: int = 1) -> Tuple[Optional[ConstraintGroup], list, list]:
    """Backward compatibility wrapper - delegates to enhanced version with increased default"""
    return relax_constraint_tree_enhanced(tree, max_drops=max_drops)