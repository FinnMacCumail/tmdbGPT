from core.model.constraint import evaluate_constraint_tree


def passes_symbolic_filter(entity: dict, constraint_tree, registry: dict) -> bool:
    movie_id = entity.get("id")
    if not movie_id:
        return False

    ids = evaluate_constraint_tree(constraint_tree, registry)
    movie_constraints = ids.get("movie", {})

    if not movie_constraints:
        return False

    logic = getattr(constraint_tree, "logic", "AND").upper()

    if logic == "OR":
        # Union: pass if entity satisfies at least one constraint
        valid_ids = set()
        for id_set in movie_constraints.values():
            valid_ids |= id_set
    else:
        # Default to intersection (AND): must satisfy all constraints
        valid_ids = None
        for id_set in movie_constraints.values():
            if valid_ids is None:
                valid_ids = set(id_set)
            else:
                valid_ids &= id_set

    return bool(valid_ids) and movie_id in valid_ids
