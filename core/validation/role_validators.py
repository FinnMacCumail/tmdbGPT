# 🧩 NEW: Flexible Role Validation Function
from typing import Dict, List, Optional, Any


def validate_roles(
    credits: Dict,
    query_entities: List[Dict],
    *,
    movie: Optional[Dict] = None,
    state: Optional[Any] = None
) -> Dict[str, bool]:
    """
    Validate whether cast/crew roles in credits satisfy the query entities.

    Args:
        credits (Dict): TMDB credits response containing 'cast' and 'crew'.
        query_entities (List[Dict]): Structured entities with roles and IDs.
        movie (Optional[Dict]): The result item to attach _provenance (if any).
        state (Optional[Any]): App state to update satisfied_roles (if present).

    Returns:
        Dict[str, bool]: Mapping of role_id keys (e.g., 'director_1032') to validation status.
    """
    cast_list = credits.get("cast", [])
    crew_list = credits.get("crew", [])

    role_results: Dict[str, bool] = {}

    for entity in query_entities:
        if entity.get("type") != "person":
            continue

        role = entity.get("role", "actor").lower()
        person_id = entity.get("resolved_id")
        person_name = entity.get("name", "").lower()

        key = f"{role}_{person_id}"

        def match_person(group: List[Dict], job_filter=None) -> bool:
            for member in group:
                if job_filter and member.get("job", "").lower() != job_filter:
                    continue
                if person_id and person_id == member.get("id"):
                    return True
                if person_name and person_name in member.get("name", "").lower():
                    return True
            return False

        if role in {"cast", "actor"}:
            passed = match_person(cast_list)
        elif role == "director":
            passed = match_person(crew_list, job_filter="director")
        elif role == "writer":
            passed = any(
                (person_id == c.get("id")
                    or person_name in c.get("name", "").lower())
                and c.get("job", "").lower() in {"writer", "screenplay"}
                for c in crew_list
            )
        elif role == "producer":
            passed = any(
                (person_id == c.get("id")
                    or person_name in c.get("name", "").lower())
                and "producer" in c.get("job", "").lower()
                for c in crew_list
            )
        elif role == "composer":
            passed = any(
                (person_id == c.get("id")
                    or person_name in c.get("name", "").lower())
                and any(k in c.get("job", "").lower() for k in ["composer", "music", "score"])
                for c in crew_list
            )
        else:
            passed = False

        role_results[key] = passed

    # Extract all satisfied role keys
    satisfied_roles = {k for k, v in role_results.items() if v}

    # 🔄 Update state
    if state and hasattr(state, "satisfied_roles"):
        state.satisfied_roles.update(satisfied_roles)

    # 🧠 Inject into provenance if available
    if movie is not None:
        movie["_provenance"] = movie.get("_provenance", {})
        movie["_provenance"]["satisfied_roles"] = list(satisfied_roles)

    # 🧪 Optional: logging comparison
    expected_roles = {
        f"{entity.get('role', 'actor').lower()}_{entity.get('resolved_id')}"
        for entity in query_entities if entity.get("type") == "person"
    }

    # print("🎯 Role Validation Summary:")
    # print(f"    ➤ Expected:  {sorted(expected_roles)}")
    # print(f"    ➤ Satisfied: {sorted(satisfied_roles)}")

    return role_results


def score_role_validation(role_results: Dict[str, bool]) -> float:
    """
    Given a dict of role validation results, return a score [0.0–1.0].
    Each role passed contributes equally.
    """
    if not role_results:
        return 0.0

    num_passed = sum(1 for v in role_results.values() if v)
    total_roles = len(role_results)
    score = num_passed / total_roles
    return round(score, 2)

 # ✅ Return True if all required actor IDs are present in the movie/TV cast.
# Used to validate cast constraints (with_people) during post-validation.


def has_all_cast(credits: Dict, required_ids: List[int]) -> bool:
    cast_ids = {c["id"] for c in credits.get("cast", [])}
    return all(pid in cast_ids for pid in required_ids)

# ✅ Return True if the specified director is present in the crew list with job="Director".
# Used to validate director-role constraints after fetching /movie/{id}/credits


def has_director(credits: Dict, director_name: str) -> bool:
    crew = credits.get("crew", [])
    return any(
        member["job"] == "Director" and member["name"].lower(
        ) == director_name.lower()
        for member in crew
    )

# Phase 20 Role-Aware Multi-Entity Planning and Execution


def has_writer(credits: Dict, writer_name: str) -> bool:
    crew = credits.get("crew", [])
    return any(
        member["job"].lower() in {
            "writer", "screenplay"} and member["name"].lower() == writer_name.lower()
        for member in crew
    )


def has_producer(credits: Dict, producer_name: str) -> bool:
    crew = credits.get("crew", [])
    return any(
        "producer" in member["job"].lower(
        ) and member["name"].lower() == producer_name.lower()
        for member in crew
    )


def has_composer(credits: Dict, composer_name: str) -> bool:
    crew = credits.get("crew", [])
    return any(
        member["job"].lower() in {
            "composer", "music", "score"} and member["name"].lower() == composer_name.lower()
        for member in crew
    )


# 🧩 NEW: Dynamic Role Validator Mapping - # Phase 20 Role-Aware Multi-Entity Planning and Execution
ROLE_VALIDATORS = {
    "cast": has_all_cast,
    "director": has_director,
    "writer": has_writer,
    "producer": has_producer,
    "composer": has_composer
}
