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
            ent_type = ent["type"]

            # ✅ Handle revenue constraints with sort_by mapping
            if ent_type == "revenue":
                param_key, priority = self._map_revenue_constraint(ent)
            # ✅ Handle date constraints with range support
            elif ent_type == "date":
                date_constraints = self._map_date_constraint(ent)
                constraints.extend(date_constraints)
                continue  # Skip the single constraint creation below
            # ✅ Skip rating entities - these should trigger sorting, not constraints
            elif ent_type == "rating":
                continue  # Let semantic parameter inference handle rating-based sorting
            else:
                param_key = get_param_key_for_type(ent_type, prefer="with_")
                priority = ent.get("priority", 2)

            # ✅ For revenue constraints, use sort_value as the constraint value
            if ent_type == "revenue" and param_key == "sort_by":
                constraint_value = ent.get("sort_value", "revenue.asc")
            else:
                constraint_value = ent.get("id") or ent.get(
                    "resolved_id") or ent.get("name")

            c = Constraint(
                key=param_key,
                value=constraint_value,
                type_=ent_type,
                subtype=ent.get("role"),
                priority=priority,
                confidence=ent.get("confidence", 1.0),
                metadata={k: v for k, v in ent.items() if k not in {
                    "type", "name", "id", "role", "priority", "confidence"}}
            )
            constraints.append(c)

        return ConstraintGroup(constraints, logic="AND")

    def _map_revenue_constraint(self, ent: dict) -> tuple:
        """Map revenue constraints to TMDB sort parameters and priority"""
        operator = ent.get("operator", "less_than")
        
        # Map revenue constraints to sort_by parameters
        if operator in ("less_than", "less_than_equal"):
            param_key = "sort_by"
            # Use popularity.desc to get mainstream movies with revenue data first
            # Then filter by threshold - avoids thousands of $0 revenue indie films
            ent["sort_value"] = "popularity.desc"
            ent["threshold"] = int(ent.get("name", 0))
            ent["threshold_operator"] = operator
        elif operator in ("greater_than", "greater_than_equal"):
            param_key = "sort_by"
            ent["sort_value"] = "revenue.desc"
            ent["threshold"] = int(ent.get("name", 0))
            ent["threshold_operator"] = operator
        else:
            param_key = "sort_by"
            ent["sort_value"] = "popularity.desc"  # Default to popular films
            
        # Revenue constraints get higher priority (lower number = higher priority)
        priority = 1
        
        return param_key, priority
    
    def _map_date_constraint(self, ent: dict) -> list:
        """Map date constraints to TMDB date parameters with range support"""
        date_value = ent.get("name") or ent.get("value")
        priority = ent.get("priority", 3)
        
        # Handle date ranges like "1990-1999"
        if isinstance(date_value, str) and "-" in date_value and len(date_value) == 9:
            # Parse decade range
            try:
                start_year, end_year = date_value.split("-")
                if len(start_year) == 4 and len(end_year) == 4:
                    # Create date range constraints
                    start_date = f"{start_year}-01-01"
                    end_date = f"{end_year}-12-31"
                    
                    return [
                        Constraint(
                            key="primary_release_date.gte",
                            value=start_date,
                            type_="date",
                            priority=priority,
                            confidence=ent.get("confidence", 1.0),
                            metadata=ent.copy()
                        ),
                        Constraint(
                            key="primary_release_date.lte", 
                            value=end_date,
                            type_="date",
                            priority=priority,
                            confidence=ent.get("confidence", 1.0),
                            metadata=ent.copy()
                        )
                    ]
            except (ValueError, IndexError):
                pass
        
        # Handle decade formats like "2010s", "90s", "1990s"
        if isinstance(date_value, str) and date_value.endswith('s'):
            decade_str = date_value[:-1]  # Remove 's'
            try:
                if len(decade_str) == 2:  # "90s" -> 1990s
                    decade_num = int(decade_str)
                    if decade_num >= 20:  # 20s-90s = 1920s-1990s
                        start_year = 1900 + decade_num
                    else:  # 00s-19s = 2000s-2010s
                        start_year = 2000 + decade_num
                elif len(decade_str) == 4:  # "2010s" -> 2010-2019
                    start_year = int(decade_str)
                else:
                    raise ValueError("Invalid decade format")
                
                end_year = start_year + 9
                start_date = f"{start_year}-01-01"
                end_date = f"{end_year}-12-31"
                
                return [
                    Constraint(
                        key="primary_release_date.gte",
                        value=start_date,
                        type_="date",
                        priority=priority,
                        confidence=ent.get("confidence", 1.0),
                        metadata=ent.copy()
                    ),
                    Constraint(
                        key="primary_release_date.lte",
                        value=end_date,
                        type_="date",
                        priority=priority,
                        confidence=ent.get("confidence", 1.0),
                        metadata=ent.copy()
                    )
                ]
            except (ValueError, IndexError):
                pass
        
        # Handle single year or fallback
        if isinstance(date_value, str) and len(date_value) == 4 and date_value.isdigit():
            # Single year constraint
            return [Constraint(
                key="primary_release_year",
                value=date_value,
                type_="date",
                priority=priority,
                confidence=ent.get("confidence", 1.0),
                metadata=ent.copy()
            )]
        
        # Fallback to primary_release_year
        return [Constraint(
            key="primary_release_year",
            value=date_value,
            type_="date", 
            priority=priority,
            confidence=ent.get("confidence", 1.0),
            metadata=ent.copy()
        )]
