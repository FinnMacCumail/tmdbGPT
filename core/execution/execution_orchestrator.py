from hashlib import sha256
import requests

from core.execution.step_runner import StepRunner
from core.execution.trace_logger import ExecutionTraceLogger


class ExecutionOrchestrator:
    def __init__(self, base_url, headers):
        self.base_url = base_url
        self.headers = headers
        self.runner = StepRunner(base_url, headers)

    def execute(self, state):
        return self.runner.execute(state)
