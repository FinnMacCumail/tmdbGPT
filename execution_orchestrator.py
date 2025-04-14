from nlp_retriever import PostStepUpdater, PathRewriter, ResultExtractor, expand_plan_with_dependencies
import requests

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
        pending = state.plan_steps

        for step in pending:
            step_id = step.get("step_id")
            path = step.get("endpoint")
            params = step.get("parameters", {})

            # Rewrite path with resolved values if needed
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

                        # Phase 7.4: dynamically inject new steps based on newly resolved entities
                        if new_entities:
                            new_steps = expand_plan_with_dependencies(state, new_entities)
                            if new_steps:
                                print(f"🔁 Appending {len(new_steps)} new dependent step(s) to execution queue.")
                                state.plan_steps.extend(new_steps)

                    except Exception as ex:
                        print(f"⚠️ Could not parse JSON or update state: {ex}")
            except Exception as ex:
                print(f"🔥 Step {step_id} failed with exception: {ex}")
                state.error = str(ex)

        return state
