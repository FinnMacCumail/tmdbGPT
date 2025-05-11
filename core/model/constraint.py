from typing import List, Union, Dict, Optional
from copy import deepcopy
from core.entity.param_utils import get_param_key_for_type


class Constraint:
    def __init__(
        self,
        key: str,
        value: Union[str, int, float],
        type_: str,
        subtype: Optional[str] = None,
        priority: int = 2,
        confidence: float = 1.0,
        metadata: Optional[dict] = None
    ):
        self.key = key
        self.value = value
        self.type = type_
        self.subtype = subtype
        self.priority = priority
        self.confidence = confidence
        self.metadata = metadata or {}

    def is_satisfied_by(self, entity):
        val = self.value
        key = self.key
        subtype = self.subtype

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
        return (
            f"Constraint(key={self.key}, value={self.value}, type={self.type}, "
            f"subtype={self.subtype}, priority={self.priority}, confidence={self.confidence})"
        )


class ConstraintGroup:
    def __init__(self, constraints: List[Union["Constraint", "ConstraintGroup"]], logic: str = "AND"):
        self.constraints = constraints
        self.logic = logic.upper()

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
        if self.logic == "OR":
            return any(eval_item(c) for c in self.constraints)
        return False


class ConstraintBuilder:
    def build_from_query_entities(self, query_entities: list) -> ConstraintGroup:
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
