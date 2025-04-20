import unittest
from unittest.mock import patch, MagicMock
from execution_orchestrator import ExecutionOrchestrator
from copy import deepcopy

class MockAppState:
    def __init__(self):
        self.extraction_result = {
            "query_entities": [
                {"name": "Emma Stone", "type": "person", "resolved_id": 54693},
                {"name": "Warner Bros", "type": "company", "resolved_id": 174},
                {"name": "comedy", "type": "genre", "resolved_id": 35}
            ],
            "intents": ["discovery.filtered"],
            "entities": ["person", "company", "genre", "movie"]
        }
        self.resolved_entities = {
            "person_id": [54693],
            "company_id": [174],
            "genre_id": [35]
        }
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

class TestMultiEntityJoin(unittest.TestCase):
    @patch("requests.get")
    def test_person_company_genre_discovery(self, mock_get):
        orchestrator = ExecutionOrchestrator("https://api.themoviedb.org/3", headers={"Authorization": "Bearer dummy"})
        state = MockAppState()

        def mocked_tmdb_get(url, headers=None, params=None):
            mock_response = MagicMock()
            mock_response.status_code = 200
            if "credits" in url:
                mock_response.json.return_value = {
                    "cast": [{"id": 54693}],  # Emma Stone
                    "crew": [{"name": "Some Director", "job": "Director"}]
                }
            else:
                mock_response.json.return_value = {
                    "results": [
                        {"id": 77, "title": "Easy A", "overview": "A smart, funny teen comedy."}
                    ]
                }
            return mock_response

        mock_get.side_effect = mocked_tmdb_get

        # Inject plan with multiple filters
        state.plan_steps = [{
            "step_id": "step_0",
            "endpoint": "/discover/movie",
            "parameters": {
                "with_people": "54693",
                "with_companies": "174",
                "with_genres": "35"
            }
        }]

        final_state = orchestrator.execute(state)

        print("✅ Completed steps:", final_state.completed_steps)
        print("📥 Responses:", final_state.responses)
        self.assertIn("step_0", final_state.completed_steps)
        self.assertTrue(any("Easy A" in r for r in final_state.responses), "Expected movie not found")

if __name__ == "__main__":
    unittest.main()