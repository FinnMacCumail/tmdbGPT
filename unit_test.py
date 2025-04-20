import unittest
from unittest.mock import patch, MagicMock
from execution_orchestrator import ExecutionOrchestrator
from copy import deepcopy
import math

class MockAppState:
    def __init__(self):
        self.extraction_result = {
            "query_entities": [
                {"name": "Brad Pitt", "type": "person", "resolved_id": 287},
                {"name": "Christopher Nolan", "type": "person", "resolved_id": 525}
            ],
            "intents": ["discovery.filtered"],
            "entities": ["person", "movie"]
        }
        self.resolved_entities = {"person_id": [287, 525]}
        self.plan_steps = []
        self.completed_steps = []
        self.data_registry = {}
        self.responses = []

    def model_copy(self, update=None):
        clone = deepcopy(self)
        if update:
            for k, v in update.items():
                setattr(clone, k, v)
        return clone

class TestPartialScoreValidation(unittest.TestCase):
    @patch("requests.get")
    def test_director_only_validation_score_half(self, mock_get):
        orchestrator = ExecutionOrchestrator("https://api.themoviedb.org/3", headers={"Authorization": "Bearer dummy"})
        state = MockAppState()

        def mocked_tmdb_get(url, headers=None, params=None):
            mock_response = MagicMock()
            mock_response.status_code = 200
            if "credits" in url:
                mock_response.json.return_value = {
                    "cast": [{"id": 999999}],  # ❌ Wrong actor ID
                    "crew": [{"name": "Christopher Nolan", "job": "Director"}]
                }
            else:
                mock_response.json.return_value = {
                    "results": [
                        {"id": 101, "title": "Tenet", "overview": "A time-bending thriller by Christopher Nolan."}
                    ]
                }
            return mock_response

        mock_get.side_effect = mocked_tmdb_get

        state.plan_steps = [{
            "step_id": "step_0",
            "endpoint": "/discover/movie",
            "parameters": {
                "with_people": "287,525"
            }
        }]

        final_state = orchestrator.execute(state)

        validated = final_state.data_registry.get("step_0", {}).get("validated", [])
        print("🧾 Validated payload:", validated)
        for r in validated:
            print("▶️ Score for", r.get("title"), "→", r.get("final_score"))

        self.assertIn("step_0", final_state.completed_steps)
        self.assertTrue(any("Tenet" in str(r) for r in validated), "Expected movie not returned")
        self.assertTrue(all(math.isclose(r.get("final_score", 0), 0.5, rel_tol=1e-3) for r in validated), "Expected partial score of 0.5")

if __name__ == "__main__":
    unittest.main()