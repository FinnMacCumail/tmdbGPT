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
    def validate_roles(result, query_entities):
        """
        Validate if cast or director matches any person entity.
        """
        credits = result.get("credits", {})
        cast_list = credits.get("cast", [])
        crew_list = credits.get("crew", [])

        for entity in query_entities:
            if entity.get("type") == "person":
                person_name = entity.get("name", "").lower()
                role = entity.get("role", "actor")  # Default to actor

                if role == "actor":
                    if any(person_name in (member.get("name", "").lower()) for member in cast_list):
                        return True
                elif role == "director":
                    if any(person_name in (member.get("name", "").lower()) and member.get("job", "").lower() == "director" for member in crew_list):
                        return True

        return False
    
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
    
    @staticmethod
    def validate_company(result, query_entities):
        """
        Validate if production company matches.
        """
        companies = result.get("production_companies", [])

        for entity in query_entities:
            if entity.get("type") == "company":
                company_name = entity.get("name", "").lower()
                if any(company_name in (company.get("name", "").lower()) for company in companies):
                    return True

        return False

    @staticmethod
    def validate_network(result, query_entities):
        """
        Validate if network matches (for TV shows).
        """
        networks = result.get("networks", [])

        for entity in query_entities:
            if entity.get("type") == "network":
                network_name = entity.get("name", "").lower()
                if any(network_name in (network.get("name", "").lower()) for network in networks):
                    return True

        return False

    @staticmethod
    def validate_genre(result, query_entities):
        """
        Validate if genres match.
        """
        genre_ids = [genre.get("id") for genre in result.get("genres", [])]

        for entity in query_entities:
            if entity.get("type") == "genre" and entity.get("resolved_id") in genre_ids:
                return True

        return False

    @staticmethod
    def validate_result(result, query_entities):
        """
        High-level: Validate one result against all entity constraints.
        Returns True if valid, False otherwise.
        """
        valid = True

        # If query mentions cast/director → validate roles
        if any(e.get("type") == "person" for e in query_entities):
            valid = valid and PostValidator.validate_roles(result, query_entities)

        # If query mentions production company
        if any(e.get("type") == "company" for e in query_entities):
            valid = valid and PostValidator.validate_company(result, query_entities)

        # If query mentions network
        if any(e.get("type") == "network" for e in query_entities):
            valid = valid and PostValidator.validate_network(result, query_entities)

        # If query mentions genre
        if any(e.get("type") == "genre" for e in query_entities):
            valid = valid and PostValidator.validate_genre(result, query_entities)

        return valid


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
