# post_validator.py
from typing import List, Dict

class PostValidator:
    @staticmethod
    def has_all_cast(credits: Dict, required_ids: List[int]) -> bool:
        cast_ids = {c["id"] for c in credits.get("cast", [])}
        return all(pid in cast_ids for pid in required_ids)

    @staticmethod
    def has_director(credits: Dict, director_name: str) -> bool:
        crew = credits.get("crew", [])
        return any(
            member["job"] == "Director" and member["name"].lower() == director_name.lower()
            for member in crew
        )

    @staticmethod
    def meets_runtime(movie_data: Dict, min_minutes: int = None, max_minutes: int = None) -> bool:
        runtime = movie_data.get("runtime")
        if runtime is None:
            return False
        return (min_minutes is None or runtime >= min_minutes) and \
               (max_minutes is None or runtime <= max_minutes)

    @staticmethod
    def has_keywords(movie_keywords: Dict, keyword_terms: List[str]) -> bool:
        found = set(kw["name"].lower() for kw in movie_keywords.get("keywords", []))
        return any(term.lower() in found for term in keyword_terms)

    @staticmethod
    def has_genres(movie_data: Dict, genre_ids: List[int]) -> bool:
        found_ids = [g["id"] for g in movie_data.get("genres", [])]
        return any(gid in found_ids for gid in genre_ids)
