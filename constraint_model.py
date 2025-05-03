
from param_utils import get_param_key_for_type
from typing import List, Union, Dict, Set
from collections import defaultdict


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


class ConstraintBuilder:
    def build_from_query_entities(self, query_entities):
        if not query_entities:
            return ConstraintGroup([], logic="AND")

        constraints = []
        for ent in query_entities:
            c = Constraint(
                key=get_param_key_for_type(ent["type"]),
                value=ent["id"],
                type_=ent["type"],
                subtype=ent.get("role"),
                priority=ent.get("priority", 2),
                confidence=ent.get("confidence", 1.0),
                metadata={k: v for k, v in ent.items() if k not in {
                    "type", "name", "id", "role", "priority", "confidence"}}
            )
            constraints.append(c)

        return ConstraintGroup(constraints, logic="AND")


def evaluate_constraint_tree(group: ConstraintGroup, data_registry: dict) -> Dict[str, Set[int]]:
    results: List[Dict[str, Set[int]]] = []

    for node in group:
        if isinstance(node, ConstraintGroup):
            result = evaluate_constraint_tree(node, data_registry)
        else:
            id_set = data_registry.get(
                node.key, {}).get(str(node.value), set())
            result = {node.type: id_set} if id_set else {}

        results.append(result)

    merged: Dict[str, Set[int]] = defaultdict(set)

    if group.logic == "AND":
        all_types = set.intersection(*(set(r.keys()) for r in results if r))
        for t in all_types:
            intersected = set.intersection(
                *(r.get(t, set()) for r in results if t in r))
            if intersected:
                merged[t] = intersected
    else:
        for r in results:
            for t, ids in r.items():
                merged[t].update(ids)

    return dict(merged)
