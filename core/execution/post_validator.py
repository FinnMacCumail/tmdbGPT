from typing import List, Dict
from core.model.evaluator import evaluate_constraint_tree, relax_constraint_tree
# avoid circular ref if static
from core.entity.param_utils import enrich_symbolic_registry
from core.entity.symbolic_filter import passes_symbolic_filter
from core.planner.plan_validator import should_apply_symbolic_filter
import requests
from typing import Dict, List, Optional, Any
from core.validation.role_validators import validate_roles
from core.validation.role_validators import (
    has_all_cast,
    has_director,
    has_writer,
    has_producer,
    has_composer,
    ROLE_VALIDATORS
)


class PostValidator:
    @staticmethod
    def validate_results(results, state):
        """
        Fully validate all discovery results against query constraints (person + non-person).
        Uses symbolic filtering *and* actual field validation.
        """
        validated = []
        query_entities = state.extraction_result.get("query_entities", [])

        for result in results:
            if PostValidator.validate_result(result, query_entities):
                validated.append(result)
                # Debug output removed

            else:
                # Debug output removed
                pass

        return validated

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

    # may be redundant

    @staticmethod
    def validate_company(result: dict, expected_company_ids: list) -> bool:
        """
        Validate that the result includes at least one of the expected company IDs
        in its production_companies field.
        """
        companies = result.get("production_companies", [])
        result_ids = {c.get("id") for c in companies if "id" in c}
        return any(cid in result_ids for cid in expected_company_ids)

    # may be redundant
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
        # Debug output removed
        valid = True

        # If query mentions cast/director → validate roles
        if any(e.get("type") == "person" for e in query_entities):
            valid = valid and validate_roles(
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

    # 🎯 Post-validation rules used for symbolic filtering.
    # Each rule triggers a follow-up /credits API call to validate roles (cast, director).
    # - Applies only if the specified endpoint and trigger_param are matched in the step.
    # - followup_endpoint_template → where to fetch real cast/crew data.
    # - validator → function that checks for presence of required people in cast/crew.
    # - args_builder → constructs arguments for validator from the query step and app state.
    POST_VALIDATION_RULES = [
        {
            "endpoint": "/discover/movie",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/movie/{movie_id}/credits",
            "validator": has_all_cast,
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
            "validator": has_director,
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
            "validator": has_all_cast,
            "args_builder": lambda step, state: {
                "required_ids": [
                    int(p) for p in step["parameters"].get("with_people", "").split(",") if p.isdigit()
                ]
            },
            "arg_source": "credits"
        },
        {
            "endpoint": "/discover/tv",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/tv/{tv_id}/credits",
            "validator": has_director,
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
            "validator": has_writer,
            "args_builder": lambda step, state: {
                "writer_name": next((
                    e["name"] for e in state.extraction_result.get("query_entities", [])
                    if e.get("type") == "person" and e.get("role") == "writer"
                ), None)
            },
            "arg_source": "credits"
        },
        {
            "endpoint": "/discover/tv",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/tv/{tv_id}/credits",
            "validator": has_producer,
            "args_builder": lambda step, state: {
                "producer_name": next((
                    e["name"] for e in state.extraction_result.get("query_entities", [])
                    if e.get("type") == "person" and e.get("role") == "producer"
                ), None)
            },
            "arg_source": "credits"
        },
        {
            "endpoint": "/discover/tv",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/tv/{tv_id}/credits",
            "validator": has_composer,
            "args_builder": lambda step, state: {
                "composer_name": next((
                    e["name"] for e in state.extraction_result.get("query_entities", [])
                    if e.get("type") == "person" and e.get("role") == "composer"
                ), None)
            },
            "arg_source": "credits"
        }
    ]

    @staticmethod
    def run_post_validations(step, data, state):

        # 🧪 Skip symbolic post-validation for symbol-free queries.
        # If no symbolic constraints (e.g., no cast, director, genre), return results as-is.
        if not should_apply_symbolic_filter(state, step):
            return data.get("results", [])

        validated = []
        results = data.get("results", [])
        query_entities = state.extraction_result.get("query_entities", [])

        # 🧪 Check if current step matches any post-validation rule (e.g. with_people on /discover/movie)
        # If matched, fetch /movie/{id}/credits (or /tv/{id}/credits) for each result to validate cast/director roles
        for rule in PostValidator.POST_VALIDATION_RULES:
            if rule["endpoint"] in step["endpoint"] and rule["trigger_param"] in step.get("parameters", {}):
                validator = rule["validator"]
                build_args = rule["args_builder"]
                args = build_args(step, state)

                for item in results:
                    item_id = item.get("id")
                    if not item_id:
                        continue
                    # 🔧 Build URL to fetch full credits for this movie/TV show.
                    # Substitutes {movie_id} or {tv_id} into the template, then sends GET request.
                    url_template = rule["followup_endpoint_template"]
                    url = f"{state.base_url}{url_template.replace('{tv_id}', str(item_id)).replace('{movie_id}', str(item_id))}"

                    try:
                        response = requests.get(url, headers=state.headers)
                        if response.status_code != 200:
                            continue

                        result_data = response.json()
                        # 🧠 Score the result against the symbolic constraint tree.
                        # Uses cast/crew info to compute a match score [0.0–1.0].
                        # Returns score and list of matched constraints (e.g., cast, director, genre).
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
                        # ✅ If score is positive, store it and apply final symbolic filter.
                        if score > 0:
                            item["final_score"] = min(score, 1.0)

                            # 🧠 Filter against symbolic constraints before final append
                            # Only keep result if it still satisfies symbolic constraint intersection (AND logic).
                            # Also append validator tags to result provenance for explanation/debugging.
                            if passes_symbolic_filter(item, state.constraint_tree, state.data_registry):
                                validated.append(item)

                            # 🧠 Append a trace tag to the result's provenance to indicate which validator passed.
                            # This is used for explanation/debugging (e.g., passed 'has_director' validation).
                            post_validations = item.setdefault(
                                "_provenance", {}).setdefault("post_validations", [])
                            if rule["validator"].__name__ == "has_all_cast":
                                post_validations.append("has_all_cast")
                            elif rule["validator"].__name__ == "has_director":
                                post_validations.append("has_director")

                            # ✅ Symbolic enrichment before adding to validated
                            # 🧩 Index this result into the symbolic registry using its credits metadata.
                            # This enables symbolic filtering and constraint matching across roles, genres, companies, etc.
                            # e.g., updates: with_people[6193] → movie_ids, director[1032] → movie_ids
                            enrich_symbolic_registry(
                                movie=item,
                                registry=state.data_registry,
                                credits=result_data
                            )

                            validated.append(item)

                    except Exception as e:
                        # Debug output removed
                        pass
                break

        return validated or results

    # 🔍 Symbolic Post-Validation Scoring Function
    # --------------------------------------------
    # This function scores a movie or TV result against the symbolic constraints extracted from a user query.
    # It evaluates whether the result satisfies role-based, categorical, or numeric constraints such as:
    #   - Person roles (cast, director, writer, composer)
    #   - Genres (via genre_ids)
    #   - Companies or networks
    #   - Future extensions: runtime, language, certification, etc.
    #
    # Scoring uses a weighted model, where each matched constraint contributes a fixed portion to the total score.
    # - Person roles carry higher weights (e.g., cast/director = 0.4)
    # - Non-person entities like genres/companies contribute smaller weights (0.3)
    # - The score is normalized based on the total number of constraints in the query
    #
    # The function also records a rich provenance log into `movie["_provenance"]` for downstream filtering, explanation,
    # and debugging, including:
    #   - final_score: normalized match score [0.0–1.0]
    #   - matched_constraints: list of symbolic constraints that were satisfied
    #   - post_validations: tags like 'has_director', 'genre_matched'
    #   - matched_roles and matched_entities for analytical filtering
    #
    # Returns:
    #   Tuple (normalized_score: float, matched_constraints: List[str])
    @staticmethod
    def score_movie_against_query(movie, state, credits=None, **kwargs):
        """
        Score the movie against the symbolic constraint tree.
        Weights:
            - Cast/Director/Writer/Composer: 0.4 each
            - Genre: 0.3
            - Company/Network: 0.3
            - Year (if desired later): 0.2
        """
        constraint_tree = kwargs.get("constraint_tree") or (
            getattr(state, "constraint_tree", None) if state else None
        )
        if constraint_tree is None:
            raise ValueError("Missing constraint_tree in state or kwargs")

        relaxed = getattr(state, "last_dropped_constraints", [])
        # Weights per entity type
        WEIGHTS = {
            "cast": 0.4,
            "director": 0.4,
            "writer": 0.3,
            "composer": 0.3,
            "genre": 0.3,
            "company": 0.3,
            "network": 0.3,
        }

        score = 0.0
        matched_constraints = []
        matched_roles = set()
        matched_entities = set()
        post_validations = []

        flattened = list(constraint_tree.flatten())

        for constraint in flattened:
            type_ = constraint.type
            key = constraint.key
            value = str(constraint.value)       # for matched_constraints
            raw_value = constraint.value        # for comparison
            subtype = (constraint.subtype or "").lower()

            if type_ == "person":
                if not credits:
                    continue
                cast = credits.get("cast", [])
                crew = credits.get("crew", [])

                if subtype in {"cast", "actor"}:
                    if raw_value in {p.get("id") for p in cast}:
                        score += WEIGHTS.get("cast", 0)
                        matched_constraints.append(f"{key}={value}")
                        matched_roles.add("cast")
                        matched_entities.add("person")
                        post_validations.append("has_all_cast")

                elif subtype == "director":
                    if raw_value in {p.get("id") for p in crew if p.get("job") == "Director"}:
                        score += WEIGHTS.get("director", 0)
                        matched_constraints.append(f"{key}={value}")
                        matched_roles.add("director")
                        matched_entities.add("person")
                        post_validations.append("has_director")

                elif subtype == "writer":
                    writer_ids = {p.get("id") for p in crew if p.get(
                        "job", "").lower() in {"writer", "screenplay"}}
                    if raw_value in writer_ids:
                        score += WEIGHTS.get("writer", 0)
                        matched_constraints.append(f"{key}={value}")
                        matched_roles.add("writer")
                        matched_entities.add("person")
                        post_validations.append("has_writer")

                elif subtype == "composer":
                    composer_ids = {
                        p.get("id") for p in crew if "music" in p.get("job", "").lower()}
                    if raw_value in composer_ids:
                        score += WEIGHTS.get("composer", 0)
                        matched_constraints.append(f"{key}={value}")
                        matched_roles.add("composer")
                        matched_entities.add("person")
                        post_validations.append("has_composer")

            elif type_ == "genre":
                if int(raw_value) in movie.get("genre_ids", []):
                    score += WEIGHTS.get("genre", 0)
                    matched_constraints.append(f"{key}={value}")
                    matched_entities.add("genre")
                    post_validations.append("genre_matched")

            elif type_ == "company":
                if PostValidator.validate_company(movie, [raw_value]):
                    score += WEIGHTS.get("company", 0)
                    matched_constraints.append(f"{key}={value}")
                    matched_entities.add("company")
                    post_validations.append("company_matched")

            elif type_ == "network":
                if movie.get("media_type") == "tv" and PostValidator.validate_network(movie, [raw_value]):
                    score += WEIGHTS.get("network", 0)
                    matched_constraints.append(f"{key}={value}")
                    matched_entities.add("network")
                    post_validations.append("network_matched")

        total_constraints = len(flattened)
        normalized_score = round(
            score / total_constraints, 3) if total_constraints else 0.0

        movie["_provenance"] = movie.get("_provenance", {})
        movie["_provenance"].update({
            "final_score": normalized_score,
            "matched_constraints": matched_constraints,
            "relaxed_constraints": [
                f"{c.key}={c.value}" for c in relaxed
            ],
            "post_validations": post_validations,
            "matched_roles": sorted(matched_roles),
            "matched_entities": sorted(matched_entities)
        })

        # ✅ Run and inject role validation summary
        role_results = validate_roles(
            credits=credits,
            query_entities=kwargs.get("query_entities", []),
            movie=movie,
            state=state
        )

        satisfied_roles = {role for role,
                           passed in role_results.items() if passed}

        if hasattr(state, "satisfied_roles"):
            state.satisfied_roles.update(satisfied_roles)

        movie["_provenance"]["satisfied_roles"] = list(satisfied_roles)

            # Debug output removed

        return normalized_score, matched_constraints
