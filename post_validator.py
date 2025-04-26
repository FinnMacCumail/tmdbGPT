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
        """
        Validate dynamically based on entity roles extracted from query.
        Returns a dict like {"cast_ok": True, "director_ok": False, ...}
        """
        results = {}

        for ent in query_entities:
            if ent.get("type") != "person":
                continue

            role = ent.get("role", "cast")  # default to cast if missing
            name_or_id = ent.get("name") if role == "director" else ent.get("resolved_id")

            validator = PostValidator.ROLE_VALIDATORS.get(role)
            if not validator or not name_or_id:
                continue

            role_key = f"{role}_ok"
            if role == "cast":
                # Validate cast using resolved person_id
                passed = validator(credits, [name_or_id])
            elif role == "director":
                # Validate director using person's name
                passed = validator(credits, name_or_id)
            else:
                # Future roles can go here
                passed = validator(credits, name_or_id)

            results[role_key] = passed

        return results

    @staticmethod
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
    
# result_scorer.py

class ResultScorer:
    @staticmethod
    def validate_entity_matches(result, query_entities):
        """
        For a given result (movie or tv show), check which entities it satisfies.
        Returns a dict {role: passed}
        """
        validations = {}

        for ent in query_entities:
            if ent.get("type") != "person" and ent.get("type") != "network" and ent.get("type") != "company":
                continue

            role = ent.get("role", "cast")
            resolved_id = ent.get("resolved_id")
            entity_type = ent.get("type")

            if not resolved_id:
                continue

            if entity_type == "person":
                if role == "cast":
                    validations[f"cast_{resolved_id}"] = ResultScorer._validate_cast(result, resolved_id)
                elif role == "director":
                    validations[f"director_{resolved_id}"] = ResultScorer._validate_director(result, resolved_id)
                # future roles can go here
            elif entity_type == "network":
                validations[f"network_{resolved_id}"] = ResultScorer._validate_network(result, resolved_id)
            elif entity_type == "company":
                validations[f"company_{resolved_id}"] = ResultScorer._validate_company(result, resolved_id)

        return validations

    @staticmethod
    def _validate_cast(result, person_id):
        return any(cast.get("id") == person_id for cast in result.get("cast", []))

    @staticmethod
    def _validate_director(result, person_id):
        return any(crew.get("job", "").lower() == "director" and crew.get("id") == person_id for crew in result.get("crew", []))

    @staticmethod
    def _validate_network(result, network_id):
        return any(n.get("id") == network_id for n in result.get("networks", []))

    @staticmethod
    def _validate_company(result, company_id):
        return any(c.get("id") == company_id for c in result.get("production_companies", []))

    @staticmethod
    def score_matches(validations: dict):
        """
        Given a dict of {role: True/False}, return normalized match score [0.0–1.0].
        """
        if not validations:
            return 0.0

        passed = sum(1 for v in validations.values() if v)
        total = len(validations)
        return round(passed / total, 2)
