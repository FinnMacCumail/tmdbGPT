from typing import List, Dict
from core.model.evaluator import evaluate_constraint_tree, relax_constraint_tree
# avoid circular ref if static
from core.entity.param_utils import enrich_symbolic_registry


class PostValidator:
    @staticmethod
    def validate_results(results, state):
        """
        Evaluate results against the current constraint tree using symbolic matching and relaxation.
        Annotate results with provenance for matched/relaxed constraints.
        """

        validated = []

        # Step 1: Evaluate current constraints
        media_matches = evaluate_constraint_tree(
            state.constraint_tree, state.data_registry)

        print("Symbolic Matches:", media_matches)

        if not media_matches["movie"] and not media_matches["tv"]:
            print("🛑 No media matches found. Attempting relaxation...")

            relaxed_tree, dropped_constraints, reasons = relax_constraint_tree(
                state.constraint_tree)
            if not relaxed_tree:
                print("❌ Could not relax any constraints.")
                return []

            state.constraint_tree = relaxed_tree
            state.last_dropped_constraints = dropped_constraints
            state.relaxation_log.extend(reasons)

            # Retry evaluation after relaxing
            media_matches = evaluate_constraint_tree(
                state.constraint_tree, state.data_registry)

        # Step 2: Apply match scoring to results
        for result in results:
            media_type = "tv" if "first_air_date" in result else "movie"
            matched_keys = []

            for param_key, id_set in media_matches.get(media_type, {}).items():
                if str(result.get("id")) in map(str, id_set):
                    matched_keys.append(f"{param_key}={result.get('id')}")

            # Provenance tagging
            result["_provenance"] = {
                "matched_constraints": matched_keys,
                "relaxed_constraints": [
                    f"{c.key}={c.value}" for c in getattr(state, "last_dropped_constraints", [])
                ],
                "post_validations": []
            }

            if matched_keys:
                validated.append(result)

        return validated

    @staticmethod
    def has_all_cast(credits: Dict, required_ids: List[int]) -> bool:
        cast_ids = {c["id"] for c in credits.get("cast", [])}
        return all(pid in cast_ids for pid in required_ids)

    @staticmethod
    def has_director(credits: Dict, director_name: str) -> bool:
        crew = credits.get("crew", [])
        return any(
            member["job"] == "Director" and member["name"].lower(
            ) == director_name.lower()
            for member in crew
        )

    # Phase 20 Role-Aware Multi-Entity Planning and Execution
    @staticmethod
    def has_writer(credits: Dict, writer_name: str) -> bool:
        crew = credits.get("crew", [])
        return any(
            member["job"].lower() in {
                "writer", "screenplay"} and member["name"].lower() == writer_name.lower()
            for member in crew
        )

    @staticmethod
    def has_producer(credits: Dict, producer_name: str) -> bool:
        crew = credits.get("crew", [])
        return any(
            "producer" in member["job"].lower(
            ) and member["name"].lower() == producer_name.lower()
            for member in crew
        )

    @staticmethod
    def has_composer(credits: Dict, composer_name: str) -> bool:
        crew = credits.get("crew", [])
        return any(
            member["job"].lower() in {
                "composer", "music", "score"} and member["name"].lower() == composer_name.lower()
            for member in crew
        )

    # 🧩 NEW: Dynamic Role Validator Mapping - # Phase 20 Role-Aware Multi-Entity Planning and Execution
    ROLE_VALIDATORS = {
        "cast": has_all_cast.__func__,
        "director": has_director.__func__,
        "writer": has_writer.__func__,
        "producer": has_producer.__func__,
        "composer": has_composer.__func__,
    }

    @staticmethod
    def meets_runtime(movie_data: Dict, min_minutes: int = None, max_minutes: int = None) -> bool:
        runtime = movie_data.get("runtime")
        if runtime is None:
            return False
        return (min_minutes is None or runtime >= min_minutes) and \
               (max_minutes is None or runtime <= max_minutes)

    @staticmethod
    def has_keywords(movie_keywords: Dict, keyword_terms: List[str]) -> bool:
        found = set(kw["name"].lower()
                    for kw in movie_keywords.get("keywords", []))
        return any(term.lower() in found for term in keyword_terms)

    @staticmethod
    def has_genres(movie_data: Dict, genre_ids: List[int]) -> bool:
        found_ids = [g["id"] for g in movie_data.get("genres", [])]
        return any(gid in found_ids for gid in genre_ids)

    # 🧩 NEW: Flexible Role Validation Function
    @staticmethod
    def validate_roles(credits: Dict, query_entities: List[Dict]) -> Dict[str, bool]:
        role_results = {}
        cast_list = credits.get("cast", [])
        crew_list = credits.get("crew", [])

        for entity in query_entities:
            if entity.get("type") != "person":
                continue

            person_name = entity.get("name", "").lower()
            person_id = entity.get("resolved_id")
            role = entity.get("role", "actor")  # default to cast

            if role == "cast" or role == "actor":
                passed = any(
                    (person_name in (member.get("name", "").lower())
                     or person_id == member.get("id"))
                    for member in cast_list
                )
            elif role == "director":
                passed = any(
                    (person_name in (member.get("name", "").lower())
                     or person_id == member.get("id"))
                    and member.get("job", "").lower() == "director"
                    for member in crew_list
                )
            elif role == "writer":
                passed = any(
                    (person_name in (member.get("name", "").lower())
                     or person_id == member.get("id"))
                    and member.get("job", "").lower() in {"writer", "screenplay"}
                    for member in crew_list
                )
            elif role == "producer":
                passed = any(
                    (person_name in (member.get("name", "").lower())
                     or person_id == member.get("id"))
                    and "producer" in member.get("job", "").lower()
                    for member in crew_list
                )
            elif role == "composer":
                passed = any(
                    (person_name in (member.get("name", "").lower())
                     or person_id == member.get("id"))
                    and ("composer" in member.get("job", "").lower() or "music" in member.get("job", "").lower())
                    for member in crew_list
                )
            else:
                passed = False

            key = f"{role}_{person_id}"
            role_results[key] = passed

        # 🧠 Debugging: Compare expected vs. satisfied roles
        expected_roles = {
            f"{entity.get('role', 'actor')}_{entity.get('resolved_id')}" for entity in query_entities if entity.get("type") == "person"}
        satisfied_roles = {role_key for role_key,
                           passed in role_results.items() if passed}

        if satisfied_roles == expected_roles:
            print(f"✅ Role validation succeeded: {satisfied_roles}")
        else:
            print("⚠️ Role validation failed")
            print(f"    ➤ Expected roles: {expected_roles}")
            print(f"    ➤ Satisfied roles: {satisfied_roles}")

        return role_results

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
    def validate_company(result: dict, expected_company_ids: list) -> bool:
        """
        Validate that the result includes at least one of the expected company IDs
        in its production_companies field.
        """
        companies = result.get("production_companies", [])
        result_ids = {c.get("id") for c in companies if "id" in c}
        return any(cid in result_ids for cid in expected_company_ids)

    @staticmethod
    def validate_network(result: dict, expected_network_ids: list) -> bool:
        """
        Validate that the result includes at least one of the expected network IDs
        in its networks field (TV only).
        """
        networks = result.get("networks", [])
        result_ids = {n.get("id") for n in networks if "id" in n}
        return any(nid in result_ids for nid in expected_network_ids)

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
            valid = valid and PostValidator.validate_roles(
                result, query_entities)

        # If query mentions production company
        if any(e.get("type") == "company" for e in query_entities):
            valid = valid and PostValidator.validate_company(
                result, query_entities)

        # If query mentions network
        if any(e.get("type") == "network" for e in query_entities):
            valid = valid and PostValidator.validate_network(
                result, query_entities)

        # If query mentions genre
        if any(e.get("type") == "genre" for e in query_entities):
            valid = valid and PostValidator.validate_genre(
                result, query_entities)

        return valid

    @staticmethod
    def validate_genres(result: dict, expected_genre_ids: list) -> bool:
        """
        Validate that the result contains at least one of the expected genres.
        """
        if not expected_genre_ids:
            return True  # No genre constraint to validate

        genres = result.get("genre_ids", []) or []
        return any(str(genre_id) in map(str, genres) for genre_id in expected_genre_ids)

    @staticmethod
    def validate_year(result: dict, expected_year: str) -> bool:
        """
        Validate that the result was released in the expected year.
        """
        if not expected_year:
            return True  # No year constraint

        date_fields = ["release_date", "first_air_date"]

        for field in date_fields:
            date_value = result.get(field)
            if date_value and date_value.startswith(expected_year):
                return True

        return False

    POST_VALIDATION_RULES = [
        {
            "endpoint": "/discover/movie",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/movie/{movie_id}/credits",
            "validator": has_all_cast.__func__,
            "args_builder": lambda step, state: {
                "required_ids": [
                    int(p) for p in step["parameters"].get("with_people", "").split(",") if p.isdigit()
                ]
            },
            "arg_source": "credits"
        },
        {
            "endpoint": "/discover/movie",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/movie/{movie_id}/credits",
            "validator": has_director.__func__,
            "args_builder": lambda step, state: {
                "director_name": next((
                    e["name"] for e in state.extraction_result.get("query_entities", [])
                    if e.get("type") == "person" and e.get("role") == "director"
                ), None)
            },
            "arg_source": "credits"
        },
        {
            "endpoint": "/discover/tv",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/tv/{tv_id}/credits",
            "validator": has_all_cast.__func__,
            "args_builder": lambda step, state: {
                "required_ids": [
                    int(p) for p in step["parameters"].get("with_people", "").split(",") if p.isdigit()
                ]
            },
            "arg_source": "credits"
        }
    ]

    @staticmethod
    def run_post_validations(step, data, state):
        validated = []
        results = data.get("results", [])
        query_entities = state.extraction_result.get("query_entities", [])

        for rule in PostValidator.POST_VALIDATION_RULES:
            if rule["endpoint"] in step["endpoint"] and rule["trigger_param"] in step.get("parameters", {}):
                validator = rule["validator"]
                build_args = rule["args_builder"]
                args = build_args(step, state)

                for item in results:
                    item_id = item.get("id")
                    if not item_id:
                        continue

                    url_template = rule["followup_endpoint_template"]
                    url = f"{state.base_url}{url_template.replace('{tv_id}', str(item_id)).replace('{movie_id}', str(item_id))}"

                    try:
                        import requests
                        response = requests.get(url, headers=state.headers)
                        if response.status_code != 200:
                            continue

                        result_data = response.json()
                        score_tuple = PostValidator.score_movie_against_query(
                            movie=item,
                            state=state,
                            credits=result_data,
                            step=step,
                            query_entities=query_entities
                        )

                        if not score_tuple:
                            continue

                        score, matched = score_tuple
                        if score > 0:
                            item["final_score"] = min(score, 1.0)

                            post_validations = item.setdefault(
                                "_provenance", {}).setdefault("post_validations", [])
                            if rule["validator"].__name__ == "has_all_cast":
                                post_validations.append("has_all_cast")
                            elif rule["validator"].__name__ == "has_director":
                                post_validations.append("has_director")

                            # ✅ Symbolic enrichment before adding to validated
                            enrich_symbolic_registry(
                                movie=item,
                                registry=state.data_registry,
                                credits=result_data
                            )

                            validated.append(item)

                    except Exception as e:
                        print(f"⚠️ Validation failed for ID={item_id}: {e}")

                break

        return validated or results

    @staticmethod
    def score_movie_against_query(movie, state, credits=None, **kwargs):
        satisfied_roles = set()
        matched_constraints = []
        relaxed = getattr(state, "last_dropped_constraints", [])

        for constraint in state.constraint_tree.flatten():
            if constraint.type == "person":
                role = constraint.subtype or "cast"
                if role == "cast":
                    if credits:
                        cast_ids = {str(p["id"])
                                    for p in credits.get("cast", [])}
                        if str(constraint.value) in cast_ids:
                            matched_constraints.append(
                                f"{constraint.key}={constraint.value}")
                            satisfied_roles.add("cast")
                        else:
                            return 0, []
                    else:
                        return 0, []
                elif role == "director":
                    if credits:
                        directors = [p for p in credits.get(
                            "crew", []) if p.get("job") == "Director"]
                        if any(str(p["id"]) == str(constraint.value) for p in directors):
                            matched_constraints.append(
                                f"{constraint.key}={constraint.value}")
                            satisfied_roles.add("director")
                        else:
                            return 0, []
                    else:
                        return 0, []

        for constraint in state.constraint_tree.flatten():
            if constraint.type == "genre":
                if int(constraint.value) in movie.get("genre_ids", []):
                    matched_constraints.append(
                        f"{constraint.key}={constraint.value}")

            if constraint.type == "company":
                if not PostValidator.validate_company(movie, [constraint.value]):
                    return 0, []
                matched_constraints.append(
                    f"{constraint.key}={constraint.value}")

            if movie.get("media_type") == "tv" and constraint.type == "network":
                if not PostValidator.validate_network(movie, [constraint.value]):
                    return 0, []
                matched_constraints.append(
                    f"{constraint.key}={constraint.value}")

        movie["_provenance"] = movie.get("_provenance", {})
        movie["_provenance"]["matched_constraints"] = matched_constraints
        movie["_provenance"]["relaxed_constraints"] = [
            f"Dropped '{c.key}={c.value}' (priority={c.priority}, confidence={c.confidence})"
            for c in relaxed
        ]

        post_validations = []
        if credits:
            cast_ids = {m.get("id") for m in credits.get("cast", [])}
            crew = credits.get("crew", [])
            director_ids = {p.get("id")
                            for p in crew if p.get("job") == "Director"}

            for constraint in state.constraint_tree.flatten():
                if constraint.type == "person" and constraint.subtype == "actor":
                    if int(constraint.value) in cast_ids:
                        post_validations.append("has_all_cast")
                        satisfied_roles.add("cast")
                if constraint.type == "person" and constraint.subtype == "director":
                    if int(constraint.value) in director_ids:
                        post_validations.append("has_director")
                        satisfied_roles.add("director")

        for constraint in state.constraint_tree.flatten():
            if constraint.type == "company":
                post_validations.append("company_matched")
            if constraint.type == "network":
                post_validations.append("network_matched")

        movie["_provenance"]["post_validations"] = post_validations
        movie["_provenance"]["satisfied_roles"] = list(satisfied_roles)

        return len(matched_constraints), matched_constraints
