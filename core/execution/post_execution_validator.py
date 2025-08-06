from typing import List, Dict
from core.model.evaluator import evaluate_constraint_tree, relax_constraint_tree
# Delegation module to avoid circular import
# Use only for importing `validate_results` from post_validator.py


class PostExecutionValidator:
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


        if not media_matches["movie"] and not media_matches["tv"]:

            relaxed_tree, dropped_constraints, reasons = relax_constraint_tree(
                state.constraint_tree)
            if not relaxed_tree:
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
