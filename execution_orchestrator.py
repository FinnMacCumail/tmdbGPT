from nlp_retriever import PostStepUpdater, PathRewriter, ResultExtractor, expand_plan_with_dependencies
import requests
from hashlib import sha256

class ExecutionOrchestrator:
    def __init__(self, base_url, headers):
        from dependency_manager import DependencyManager
        self.dependency_manager = DependencyManager()
        self.base_url = base_url
        self.headers = headers

    def execute(self, state):
        state.error = None
        state.data_registry = {}
        state.completed_steps = []
        seen_step_keys = set()
        step_origin_depth = {}
        MAX_CHAIN_DEPTH = 3

        i = 0
        while i < len(state.plan_steps):
            step = state.plan_steps[i]
            i += 1

            param_string = "&".join(f"{k}={v}" for k, v in sorted(step.get("parameters", {}).items()))
            dedup_key = f"{step['endpoint']}?{param_string}"
            step_hash = sha256(dedup_key.encode()).hexdigest()

            if step_hash in seen_step_keys:
                print(f"🔁 Skipping duplicate step: {step['endpoint']} with same parameters")
                continue

            seen_step_keys.add(step_hash)
            step_id = step.get("step_id")
            depth = step_origin_depth.get(step_id, 0)
            if depth > MAX_CHAIN_DEPTH:
                print(f"🔁 Loop suppression: skipping step {step_id} (depth={depth})")
                continue

            path = step.get("endpoint")
            params = step.get("parameters", {})

            for k, v in params.items():
                if f"{{{k}}}" in path:
                    path = path.replace(f"{{{k}}}", str(v))

            path = PathRewriter.rewrite(path, state.resolved_entities)
            full_url = f"{self.base_url}{path}"
            print(f"\n⚡ Executing {step_id}: {path}")

            try:
                response = requests.get(full_url, headers=self.headers, params=params)
                if response.status_code == 200:
                    print(f"✅ Success: {response.status_code}")
                    try:
                        json_data = response.json()
                        state.data_registry[step_id] = json_data

                        previous_entities = set(state.resolved_entities.keys())
                        state = PostStepUpdater.update(state, step, json_data)
                        new_entities = {
                            k: v for k, v in state.resolved_entities.items()
                            if k not in previous_entities
                        }

                        summaries = ResultExtractor.extract(step["endpoint"], json_data)
                        if summaries:
                            state.responses.extend(summaries)
                            ExecutionTraceLogger.log_step(step_id, path, "Success", summaries[0] if summaries else "<no summary>")

                        if new_entities:
                            new_steps = expand_plan_with_dependencies(state, new_entities)
                            if new_steps:
                                print(f"🔁 Appending {len(new_steps)} new dependent step(s) to execution queue.")
                                for new_step in new_steps:
                                    state.plan_steps.append(new_step)
                                    step_origin_depth[new_step["step_id"]] = depth + 1

                    except Exception as ex:
                        print(f"⚠️ Could not parse JSON or update state: {ex}")
            except Exception as ex:
                print(f"🔥 Step {step_id} failed with exception: {ex}")
                ExecutionTraceLogger.log_step(step_id, path, f"Failed ({str(ex)})")
                state.error = str(ex)

        return state

class ExecutionTraceLogger:
    @staticmethod
    def log_step(step_id, path, status, summary=None):
        print("\n📍 Execution Trace")
        print(f"├─ Step: {step_id}")
        print(f"├─ Endpoint: {path}")
        print(f"├─ Status: {status}")
        if summary:
            print(f"└─ Result: {summary[:100]}{'...' if len(summary) > 100 else ''}")

# Usage inside orchestrator loop:
# After each response:
# will be moved inside try block where 'summaries' is defined

# On failure:
# will be moved inside exception block where 'response' is defined