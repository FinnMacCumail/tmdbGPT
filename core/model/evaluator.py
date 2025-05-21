from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple, Dict, Set

from core.model.constraint import Constraint, ConstraintGroup


def evaluate_constraint_tree(group: ConstraintGroup, data_registry: dict) -> Dict[str, Dict[str, Set[int]]]:

    if not group or not group.constraints:
        # ✅ No constraints = everything passes
        return {"movie": {}, "tv": {}}

    results = []

    for node in group:
        if isinstance(node, ConstraintGroup):
            result = evaluate_constraint_tree(node, data_registry)
        else:
            value_str = str(node.value)
            media_type = node.metadata.get("media_type", "movie")
            id_set = data_registry.get(node.key, {}).get(value_str, set())
            result = {media_type: {node.key: id_set}} if id_set else {}
        results.append(result)

    # Always return a consistent structure
    merged = {"movie": {}, "tv": {}}

    if not results:
        return merged

    if group.logic == "AND":
        all_sets = []
        for r in results:
            for media, keys in r.items():
                for id_set in keys.values():
                    if id_set:
                        all_sets.append(id_set)

        if not all_sets:
            return merged

        global_intersection = set.intersection(*all_sets)

        for r in results:
            for media, keys in r.items():
                for k, v in keys.items():
                    filtered = v & global_intersection
                    if filtered:
                        merged[media].setdefault(k, set()).update(filtered)

    elif group.logic == "OR":
        for r in results:
            for media, keys in r.items():
                for k, v in keys.items():
                    merged[media].setdefault(k, set()).update(v)

    return merged


def relax_constraint_tree(
    tree: ConstraintGroup,
    max_drops: int = 1
) -> Tuple[Optional[ConstraintGroup], list, list]:
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

    domain_priority = {
        "company": 1,
        "network": 1,
        "genre": 2,
        "date": 3,
        "language": 4,
        "runtime": 5,
        "person": 6
    }

    sorted_constraints = sorted(
        flat_constraints,
        key=lambda c: (domain_priority.get(c.type, 9),
                       c.priority, -c.confidence)
    )

    dropped = []
    reasons = []

    for constraint in sorted_constraints:
        if len(dropped) >= max_drops:
            break

        def remove_constraint(group):
            group.constraints = [
                c for c in group.constraints
                if not (not isinstance(c, ConstraintGroup) and c.key == constraint.key and c.value == constraint.value)
            ]
            for c in group.constraints:
                if isinstance(c, ConstraintGroup):
                    remove_constraint(c)

        remove_constraint(relaxed)
        dropped.append(constraint)
        reasons.append(
            f"Dropped '{constraint.key}={constraint.value}' (type={constraint.type}, "
            f"priority={constraint.priority}, confidence={constraint.confidence})"
        )

    if not dropped:
        return None, [], ["No constraints could be dropped"]

    return relaxed, dropped, reasons
