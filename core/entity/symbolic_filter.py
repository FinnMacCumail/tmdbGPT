from core.model.evaluator import evaluate_constraint_tree

import requests
from core.entity.param_utils import update_symbolic_registry


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
    Filter a list of TMDB entities (movies or TV) based on symbolic constraint satisfaction.

    Args:
        entities (list): List of TMDB media dicts (must include 'id').
        constraint_tree: ConstraintGroup holding current query constraints.
        registry (dict): Enriched symbolic registry (e.g. state.data_registry).

    Returns:
        list: Entities that pass the symbolic filter.
    """
    ids = evaluate_constraint_tree(constraint_tree, registry)

    # Support both movie and tv entries
    valid_ids = set()
    for media_type in ("movie", "tv"):
        matches = ids.get(media_type, {})
        if not matches:
            continue

        logic = getattr(constraint_tree, "logic", "AND").upper()
        if logic == "OR":
            for match_set in matches.values():
                valid_ids |= match_set
        else:
            intersection = None
            for match_set in matches.values():
                if intersection is None:
                    intersection = set(match_set)
                else:
                    intersection &= match_set
            if intersection:
                valid_ids |= intersection

    filtered = []
    for m in entities:
        mid = m.get("id")
        passed = mid in valid_ids
        print(
            f"🔍 {m.get('title') or m.get('name')} — passed symbolic filter: {passed}")
        if passed:
            filtered.append(m)

    return filtered


def lazy_enrich_and_filter(entity, constraint_tree, registry, headers, base_url) -> bool:
    """
    Try symbolic filtering first. If it fails, enrich the entity and try again.
    Returns True if the entity passes symbolic filtering after optional enrichment.
    """
    entity_id = entity.get("id")
    media_type = "tv" if "first_air_date" in entity else "movie"

    if passes_symbolic_filter(entity, constraint_tree, registry):
        return True  # already passes

    # 🛠 Fetch full entity details
    try:
        url = f"{base_url}/{media_type}/{entity_id}"
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            return False
        enriched = res.json()
        # Patch the fallback entity with enriched fields
        for field in ["genres", "production_companies", "networks", "original_language", "origin_country"]:
            if field in enriched:
                entity[field] = enriched[field]

        # Optionally populate `genre_ids`, etc.
        if "genres" in enriched:
            entity["genre_ids"] = [g["id"] for g in enriched.get("genres", [])]
        if "networks" in enriched:
            entity["network_ids"] = [n["id"]
                                     for n in enriched.get("networks", [])]
        if "production_companies" in enriched:
            entity["production_company_ids"] = [c["id"]
                                                for c in enriched.get("production_companies", [])]
        if "original_language" in enriched:
            entity["original_language"] = enriched.get("original_language")
        if "origin_country" in enriched:
            entity["origin_country"] = enriched.get("origin_country")

        # Symbolically enrich registry after patching
        update_symbolic_registry(entity, registry)

        # Retry filtering
        return passes_symbolic_filter(entity, constraint_tree, registry)

    except Exception as e:
        print(f"⚠️ Lazy enrichment failed for {entity_id}: {e}")
        return False
