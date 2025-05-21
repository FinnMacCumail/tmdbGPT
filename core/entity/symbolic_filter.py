from core.model.evaluator import evaluate_constraint_tree

import requests
from core.entity.param_utils import update_symbolic_registry


"""
Check if a given movie or TV entity satisfies all symbolic constraints
by evaluating whether its ID appears in the intersection (AND) or union (OR)
of constraint-matching ID sets from the registry.

note!! - It only looks at "movie" and ignores "tv". - needs refactoring
"""


def passes_symbolic_filter(entity: dict, constraint_tree, registry: dict) -> bool:
    """
    Check if an entity (e.g., movie or TV show) passes the symbolic constraint filter.

    Args:
        entity (dict): A TMDB entity with at least an 'id' field.
        constraint_tree: ConstraintGroup instance representing current query constraints.
        registry (dict): Data registry populated via enrich_symbolic_registry(...).

    Returns:
        bool: True if the entity satisfies the symbolic constraint tree.
    """
    entity_id = entity.get("id")
    if not entity_id:
        return False

    if (
        not constraint_tree or
        not getattr(constraint_tree, "constraints", None) or
        len(constraint_tree.constraints) == 0
    ):
        # ✅ No constraints = allow all
        print("✅ passes_symbolic_filter bypassed — no constraints to check.")
        return True

    constraint_ids = evaluate_constraint_tree(constraint_tree, registry)

    # 🧠 Determine media type based on release info
    if "first_air_date" in entity:
        media_type = "tv"
    else:
        media_type = "movie"

    constraint_sets = constraint_ids.get(media_type, {})
    if not constraint_sets:
        return False

    logic_mode = getattr(constraint_tree, "logic", "AND").upper()

    if logic_mode == "OR":
        # ✅ Pass if the entity appears in any constraint match
        for match_set in constraint_sets.values():
            if entity_id in match_set:
                return True
        return False

    # ✅ Default: must appear in the intersection of all constraint match sets
    valid_ids = None
    for match_set in constraint_sets.values():
        if valid_ids is None:
            valid_ids = set(match_set)
        else:
            valid_ids &= match_set

    return bool(valid_ids) and entity_id in valid_ids


def lazy_enrich_and_filter(entity, constraint_tree, registry, headers, base_url) -> bool:
    """
    Attempt symbolic filtering on an entity. If it fails, fetch enriched details and retry.

    Args:
        entity (dict): TMDB entity (movie or TV).
        constraint_tree (ConstraintGroup): Current symbolic constraints.
        registry (dict): Symbolic data registry (e.g., state.data_registry).
        headers (dict): HTTP headers for API calls.
        base_url (str): Base TMDB API URL.

    Returns:
        bool: True if entity passes symbolic filter (initially or after enrichment).
    """
    entity_id = entity.get("id")
    if not entity_id:
        print("⚠️ Entity is missing an ID — cannot enrich.")
        return False

    media_type = "tv" if "first_air_date" in entity else "movie"

    # ✅ Shortcut: no constraints, always pass
    if not constraint_tree or not getattr(constraint_tree, "constraints", []):
        print(
            f"✅ No constraints — accepted: {entity.get('title') or entity.get('name')}")
        return True

    # ✅ Initial filter pass
    if passes_symbolic_filter(entity, constraint_tree, registry):
        return True

    # 🛠 Attempt enrichment if initial filter fails
    try:
        url = f"{base_url}/{media_type}/{entity_id}"
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            print(
                f"⚠️ Failed to fetch enrichment for ID={entity_id} — status: {res.status_code}")
            return False

        enriched = res.json()

        # 🔄 Merge enrichable fields into the original entity
        patch_fields = {
            "genres": lambda v: [g["id"] for g in v],
            "production_companies": lambda v: [c["id"] for c in v],
            "networks": lambda v: [n["id"] for n in v],
            "original_language": lambda v: v,
            "origin_country": lambda v: v
        }

        for key, transform in patch_fields.items():
            if key in enriched:
                entity[key] = enriched[key]
                # Also create `_ids` versions where relevant
                if key in ["genres", "production_companies", "networks"]:
                    entity[f"{key[:-1]}_ids"] = transform(enriched[key])

        # ♻️ Update registry after enrichment
        update_symbolic_registry(entity, registry)

        # 🔁 Retry symbolic filter
        return passes_symbolic_filter(entity, constraint_tree, registry)

    except Exception as e:
        print(f"⚠️ Lazy enrichment failed for ID={entity_id}: {e}")
        return False
