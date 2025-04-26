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

    # 🧩 NEW: Dynamic Role Validator Mapping
    ROLE_VALIDATORS = {
        "cast": has_all_cast.__func__,
        "director": has_director.__func__,
        # Placeholder: if you add more validators (writer, composer, producer), map them here
    }

    # 🧩 NEW: Flexible Role Validation Function
    @staticmethod
    def validate_roles(credits: Dict, query_entities: List[Dict]) -> Dict[str, bool]:
        results = {}

        for ent in query_entities:
            if ent.get("type") != "person":
                continue
            role = ent.get("role", "cast")  # default to cast if missing
            name_or_id = ent.get("name") if role == "director" else ent.get("resolved_id")

            validator = PostValidator.ROLE_VALIDATORS.get(role)
            if not validator or not name_or_id:
                continue

            if role == "cast":
                if "cast_ok" not in results:
                    results["cast_ok"] = validator(credits, [name_or_id])
                else:
                    results["cast_ok"] = results["cast_ok"] and validator(credits, [name_or_id])
            elif role == "director":
                if "director_ok" not in results:
                    results["director_ok"] = validator(credits, name_or_id)
                else:
                    results["director_ok"] = results["director_ok"] or validator(credits, name_or_id)
            else:
                role_key = f"{role}_ok"
                results[role_key] = validator(credits, name_or_id)

        return results
