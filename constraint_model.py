
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

    def is_satisfied_by(self, entity):
        val = self.value
        key = self.key
        subtype = getattr(self, "subtype", None)

        if key == "with_people":
            role = "crew" if subtype == "director" else "cast"
            people = entity.get("credits", {}).get(role, [])
            people_ids = [p["id"] for p in people if isinstance(p, dict)]
            return val in people_ids

        elif key == "with_genres":
            return val in entity.get("genre_ids", [])

        elif key == "with_companies":
            return val in [c["id"] for c in entity.get("production_companies", [])]

        elif key == "with_networks":
            return val in [n["id"] for n in entity.get("networks", [])]

        elif key == "primary_release_year":
            return str(val) in entity.get("release_date", "")

        else:
            actual = entity.get(key)
            return val == actual or (isinstance(actual, list) and val in actual)

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

    def is_satisfied_by(self, entity, log=None):
        def eval_item(item):
            res = item.is_satisfied_by(entity) if isinstance(
                item, Constraint) else item.is_satisfied_by(entity, log)
            if log is not None and isinstance(item, Constraint):
                log.append((item, res))
            return res

        if self.logic == "AND":
            return all(eval_item(c) for c in self.constraints)
        elif self.logic == "OR":
            return any(eval_item(c) for c in self.constraints)
        return False


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


def evaluate_constraint_tree(group: ConstraintGroup, data_registry: dict) -> Dict[str, Dict[str, Set[int]]]:
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

    # phase 22 - GPMJE+
    merged = {"movie": defaultdict(set), "tv": defaultdict(set)}

    if not results:
        return {}

    if group.logic == "AND":
        all_sets = []
        for r in results:
            for media, keys in r.items():
                for id_set in keys.values():
                    if id_set:
                        all_sets.append(id_set)

        if not all_sets:
            return {}
        global_intersection = set.intersection(*all_sets)

        for r in results:
            for media, keys in r.items():
                for k, v in keys.items():
                    filtered = v & global_intersection
                    if filtered:
                        merged[media][k].update(filtered)

    elif group.logic == "OR":
        for r in results:
            for media, keys in r.items():
                for k, v in keys.items():
                    merged[media][k].update(v)

    return merged

# phase 22 - Priority-based relaxation across domains


def relax_constraint_tree(
    tree: ConstraintGroup,
    max_drops: int = 1
) -> Tuple[Optional[ConstraintGroup], List[Constraint], List[str]]:
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
        removed = False

        def remove_constraint(group):
            nonlocal removed
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
            f"Dropped '{constraint.key}={constraint.value}' (type={constraint.type}, priority={constraint.priority}, confidence={constraint.confidence})")

    if not dropped:
        return None, [], ["No constraints could be dropped"]

    return relaxed, dropped, reasons
