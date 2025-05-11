# core/llm/role_inference.py

ROLE_ALIASES = {
    "directed by": "director",
    "produced by": "producer",
    "written by": "writer",
    "screenplay by": "writer",
    "script by": "writer",
    "music by": "composer",
    "scored by": "composer",
    "starring": "cast",
    "acted in": "cast",
    "featuring": "cast",
    "performance by": "cast",
}


def infer_role_from_query(query: str) -> str:
    query_lower = query.lower()
    for phrase, role in ROLE_ALIASES.items():
        if phrase in query_lower:
            return role
    return "cast"  # default fallback


def infer_role_for_entity(name: str, query: str) -> str:
    query_lower = query.lower()
    name_lower = name.lower()

    idx = query_lower.find(name_lower)
    if idx == -1:
        return "cast"

    window_before = 50
    window_after = 50

    start_before = max(0, idx - window_before)
    end_before = idx

    start_after = idx + len(name_lower)
    end_after = min(len(query_lower), start_after + window_after)

    context_before = query_lower[start_before:end_before]
    context_after = query_lower[start_after:end_after]

    role_priority = [
        ("starring", "cast"),
        ("acted in", "cast"),
        ("performance by", "cast"),
        ("featuring", "cast"),
        ("directed by", "director"),
        ("produced by", "producer"),
        ("written by", "writer"),
        ("screenplay by", "writer"),
        ("script by", "writer"),
        ("music by", "composer"),
        ("scored by", "composer"),
    ]

    for phrase, role in role_priority:
        if phrase in context_before:
            return role
    for phrase, role in role_priority:
        if phrase in context_after:
            return role

    return "cast"


def infer_media_type_from_query(query: str) -> str:
    query_lower = query.lower()
    if "tv show" in query_lower or "series" in query_lower:
        return "tv"
    elif "movie" in query_lower or "film" in query_lower:
        return "movie"
    else:
        return "both"
