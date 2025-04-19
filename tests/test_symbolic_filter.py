import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from param_utils import is_entity_compatible, is_intent_supported


def test_entity_compatibility():
    assert is_entity_compatible(set(), []) == True
    assert is_entity_compatible({"company_id"}, ["with_companies"]) == True
    assert is_entity_compatible({"person_id"}, ["with_companies"]) == False
    assert is_entity_compatible({"movie_id"}, ["with_people"]) == False

def test_intent_filtering():
    assert is_intent_supported("discovery.filtered", ["discovery.filtered"]) is True
    assert is_intent_supported("trending.popular", ["search.multi", "details.movie"]) is False
    assert is_intent_supported("", ["details.movie"]) is False