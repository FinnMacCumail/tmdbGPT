from hashlib import sha256
import requests

from core.execution.step_runner import StepRunner
from core.execution.trace_logger import ExecutionTraceLogger


def fetch_credits_for_entity(entity, base_url, headers):
    """
    Fetch and return credits for either movie or tv.
    """
    media_type = "tv" if entity.get("type", "").startswith("tv") else "movie"
    entity_id = entity.get("id")
    if not entity_id:
        return None

    try:
        url = f"{base_url}/{media_type}/{entity_id}/credits"
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(
            f"⚠️ Could not fetch credits for {media_type} ID {entity_id}: {e}")
    return None


class ExecutionOrchestrator:
    def __init__(self, base_url, headers):
        self.base_url = base_url
        self.headers = headers
        self.runner = StepRunner(base_url, headers)

    def execute(self, state):
        return self.runner.execute(state)
