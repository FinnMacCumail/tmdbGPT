
from param_utils import get_param_key_for_type
from typing import List, Union, Dict, Set, Tuple, Optional
from collections import defaultdict
from copy import deepcopy


class Constraint:
    def __init__(
        self,
        key: str,
        value: Union[str, int, float],
        type_: str,
        subtype: str = None,
        priority: int = 2,
        confidence: float = 1.0,
        metadata: dict = None
    ):
        self.key = key
        self.value = value
        self.type = type_
        self.subtype = subtype  # e.g., "director" for person-type
        self.priority = priority
        self.confidence = confidence
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "key": self.key,
            "value": self.value,
            "type": self.type,
            "subtype": self.subtype,
            "priority": self.priority,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

    def __repr__(self):
        return f"Constraint(key={self.key}, value={self.value}, type={self.type}, subtype={self.subtype}, priority={self.priority}, confidence={self.confidence})"


class ConstraintGroup:
    def __init__(self, constraints: List[Union["Constraint", "ConstraintGroup"]], logic: str = "AND"):
        self.constraints = constraints
        self.logic = logic.upper()  # "AND" or "OR"

    def to_dict(self):
        return {
            "logic": self.logic,
            "constraints": [c.to_dict() if hasattr(c, "to_dict") else str(c) for c in self.constraints]
        }

    def __iter__(self):
        return iter(self.constraints)

    def __repr__(self):
        return f"ConstraintGroup(logic={self.logic}, constraints={self.constraints})"

    def flatten(self):
        """Recursively yields all constraints in a nested group."""
        for item in self.constraints:
            if isinstance(item, ConstraintGroup):
                yield from item.flatten()
            else:
                yield item


class ConstraintBuilder:
    def build_from_query_entities(self, query_entities):
        if not query_entities:
            return ConstraintGroup([], logic="AND")

        constraints = []
        for ent in query_entities:
            c = Constraint(
                key=get_param_key_for_type(ent["type"], prefer="with_"),
                value=ent.get("id") or ent.get("resolved_id"),
                type_=ent["type"],
                subtype=ent.get("role"),
                priority=ent.get("priority", 2),
                confidence=ent.get("confidence", 1.0),
                metadata={k: v for k, v in ent.items() if k not in {
                    "type", "name", "id", "role", "priority", "confidence"}}
            )
            constraints.append(c)

        return ConstraintGroup(constraints, logic="AND")


def evaluate_constraint_tree(group, data_registry: dict) -> Dict[str, Set[int]]:
    """
    Evaluate a ConstraintGroup recursively and return matched entity IDs
    grouped by constraint.key. AND groups return only intersecting IDs.
    """
    print(
        f"🌲 Evaluating ConstraintGroup ({group.logic}) with members: {group.constraints}")
    results: List[Dict[str, Set[int]]] = []

    for node in group:
        if isinstance(node, ConstraintGroup):
            result = evaluate_constraint_tree(node, data_registry)
        else:
            value_str = str(node.value)
            id_set = data_registry.get(node.key, {}).get(value_str, set())
            print(f"🔍 Node {node.key}={node.value} matched IDs: {id_set}")
            result = {node.key: id_set} if id_set else {}
        results.append(result)

    merged: Dict[str, Set[int]] = defaultdict(set)

    if not results:
        return {}

    if group.logic == "AND":
        # Step 1: Find global intersection of all ID sets
        all_sets = [id_set for r in results for id_set in r.values() if id_set]
        if not all_sets:
            return {}
        global_intersection = set.intersection(*all_sets)

        # Step 2: Retain only keys whose ID sets intersect with global intersection
        for r in results:
            for k, v in r.items():
                filtered = v & global_intersection
                if filtered:
                    merged[k].update(filtered)

    elif group.logic == "OR":
        for r in results:
            for k, v in r.items():
                merged[k].update(v)

    print(f"🎯 Final merged constraint results: {dict(merged)}")
    return dict(merged)


# phase 21.6 - Step 6: Logging and Trace


def relax_constraint_tree(
    tree: ConstraintGroup,
    max_drops: int = 1
) -> Tuple[ConstraintGroup, List[object], List[str]]:
    """
    Attempt to relax a constraint tree by dropping the lowest priority/confidence constraints.
    Returns:
        (relaxed_tree, dropped_constraints, relaxation_log)
    """
    print("♻️ Starting constraint relaxation...")
    relaxed = deepcopy(tree)
    flat_constraints = []

    def collect_constraints(group):
        for c in group.constraints:
            if isinstance(c, ConstraintGroup):
                yield from collect_constraints(c)
            else:
                yield c

    flat_constraints = list(collect_constraints(relaxed))

    if not flat_constraints:
        print("⚠️ No constraints available to relax.")
        return None, [], ["No constraints found in tree"]

    # Sort by (priority, confidence)
    sorted_constraints = sorted(
        flat_constraints,
        key=lambda c: (c.priority, -c.confidence)
    )

    dropped = []
    reasons = []

    for constraint in sorted_constraints:
        if len(dropped) >= max_drops:
            break
        print(
            f"💥 Dropping constraint: {constraint.key}={constraint.value} (priority={constraint.priority}, confidence={constraint.confidence})")
        removed = False

        def remove_constraint(group):
            nonlocal removed
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
        reasons.append(
            f"Dropped '{constraint.key}={constraint.value}' (priority={constraint.priority}, confidence={constraint.confidence})"
        )

    if not dropped:
        print("🛑 No constraints were dropped during relaxation.")
        return None, [], ["No constraints could be dropped"]

    print(
        f"♻️ Relaxed tree after dropping: {[(c.key, c.value) for c in dropped]}")
    return relaxed, dropped, reasons


def normalize_constraint_tree(group: ConstraintGroup) -> ConstraintGroup:
    """
    Flattens nested ConstraintGroups and deduplicates Constraints within a group.
    """
    seen = set()
    flat_constraints = []

    def flatten(group_or_constraint):
        if isinstance(group_or_constraint, ConstraintGroup):
            for member in group_or_constraint:
                flatten(member)
        elif isinstance(group_or_constraint, Constraint):
            sig = (group_or_constraint.key, group_or_constraint.value,
                   group_or_constraint.type)
            if sig not in seen:
                seen.add(sig)
                flat_constraints.append(group_or_constraint)

    flatten(group)

    return ConstraintGroup(flat_constraints, logic=group.logic)
