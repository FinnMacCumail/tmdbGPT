# test_param_utils.py

from param_utils import resolve_parameter_for_entity

def test_resolve_parameter_for_entity():

    # --- Symbolic Filters (Prefer with_* query parameters)
    person_param = resolve_parameter_for_entity("person")
    assert person_param in {"with_people", "with_cast", "with_crew"}, f"❌ Unexpected param {person_param} for 'person'"

    company_param = resolve_parameter_for_entity("company")
    assert company_param in {"with_companies", "without_companies"}, f"❌ Unexpected param {company_param} for 'company'"

    network_param = resolve_parameter_for_entity("network")
    assert network_param == "with_networks", f"❌ Unexpected param {network_param} for 'network'"

    genre_param = resolve_parameter_for_entity("genre")
    assert genre_param in {"with_genres", "without_genres"}, f"❌ Unexpected param {genre_param} for 'genre'"

    keyword_param = resolve_parameter_for_entity("keyword")
    assert keyword_param in {"with_keywords", "without_keywords"}, f"❌ Unexpected param {keyword_param} for 'keyword'"

    # --- Path Slot Fallbacks
    collection_param = resolve_parameter_for_entity("collection")
    assert collection_param == "collection_id", f"❌ Unexpected param {collection_param} for 'collection'"

    review_param = resolve_parameter_for_entity("review")
    assert review_param == "review_id", f"❌ Unexpected param {review_param} for 'review'"

    credit_param = resolve_parameter_for_entity("credit")
    assert credit_param == "credit_id", f"❌ Unexpected param {credit_param} for 'credit'"


if __name__ == "__main__":
    test_resolve_parameter_for_entity()
