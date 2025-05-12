from core.model.constraint import evaluate_constraint_tree


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

    constraint_ids = evaluate_constraint_tree(constraint_tree, registry)
    movie_constraint_sets = constraint_ids.get("movie", {})
    if not movie_constraint_sets:
        return False

    logic_mode = getattr(constraint_tree, "logic", "AND").upper()

    if logic_mode == "OR":
        # ✅ Pass if the entity appears in any constraint match
        for match_set in movie_constraint_sets.values():
            if entity_id in match_set:
                return True
        return False

    # ✅ Default: must appear in the intersection of all constraint match sets
    valid_ids = None
    for match_set in movie_constraint_sets.values():
        if valid_ids is None:
            valid_ids = set(match_set)
        else:
            valid_ids &= match_set

    return bool(valid_ids) and entity_id in valid_ids


def filter_valid_movies(entities: list, constraint_tree, registry: dict) -> list:
    """
    Filter a list of TMDB entities based on symbolic constraint satisfaction.

    Args:
        entities (list): List of TMDB media dicts (must include 'id').
        constraint_tree: ConstraintGroup holding current query constraints.
        registry (dict): Enriched symbolic registry (e.g. state.data_registry).

    Returns:
        list: Entities that pass the symbolic filter.
    """
    ids = evaluate_constraint_tree(constraint_tree, registry)
    movie_constraints = ids.get("movie", {})
    if not movie_constraints:
        return []

    logic = getattr(constraint_tree, "logic", "AND").upper()

    if logic == "OR":
        valid_ids = set()
        for match_set in movie_constraints.values():
            valid_ids |= match_set
    else:
        valid_ids = None
        for match_set in movie_constraints.values():
            if valid_ids is None:
                valid_ids = set(match_set)
            else:
                valid_ids &= match_set

    return [m for m in entities if m.get("id") in valid_ids]
