from nlp_retriever import PostStepUpdater, PathRewriter, ResultExtractor, expand_plan_with_dependencies
import requests
from hashlib import sha256
from post_validator import PostValidator
from entity_reranker import EntityAwareReranker 
from plan_validator import PlanValidator
import json

class ExecutionOrchestrator:
    
    VALIDATION_REGISTRY = [
        {
            "endpoint": "/discover/movie",
            "trigger_param": "with_people",
            "followup_endpoint_template": "/movie/{movie_id}/credits",
            "validator": PostValidator.has_all_cast,
            "args_builder": lambda step, state: {
                "required_ids": [
                    int(pid)
                    for pid in step["parameters"].get("with_people", "").split(",")
                    if pid.isdigit()
                ]
            },
            "arg_source": "credits"
        },
        {
            "endpoint": "/discover/movie",
            "trigger_param": "director_name",
            "followup_endpoint_template": "/movie/{movie_id}/credits",
            "validator": PostValidator.has_director,
            "args_builder": lambda step, state: {
                "director_name": state.extraction_result.get("query_entities", [None])[0]
            },
            "arg_source": "credits"
        }
        # Add more validations here
    ]
    
    def __init__(self, base_url, headers):
        from dependency_manager import DependencyManager
        self.dependency_manager = DependencyManager()
        self.base_url = base_url
        self.headers = headers
        self.validator = PlanValidator()

    def _run_post_validations(self, step, data, state):
        validated = []
        movie_results = data.get("results", [])
        print(f"🔍 Running post-validations on {len(movie_results)} movie(s)...")

        for rule in self.VALIDATION_REGISTRY:
            if rule["endpoint"] in step["endpoint"] and rule["trigger_param"] in step.get("parameters", {}):
                print(f"🧪 Applying validation rule: {rule['validator'].__name__}")
                validator = rule["validator"]
                build_args = rule["args_builder"]
                args = build_args(step, state)

                # Fallback assumption: two people → one cast, one director
                people_ids = step["parameters"].get("with_people", "")
                person_ids = [int(pid) for pid in people_ids.split(",") if pid.isdigit()]
                cast_id = person_ids[0] if len(person_ids) >= 1 else None

                query_entities = state.extraction_result.get("query_entities", [])
                director_name = None
                for qe in query_entities:
                    if qe["name"].lower() != state.resolved_entities.get("person_id", [])[0]:
                        director_name = qe["name"]

                for movie in movie_results:
                    movie_id = movie.get("id")
                    if not movie_id:
                        continue

                    url = f"{self.base_url}/movie/{movie_id}/credits"
                    try:
                        response = requests.get(url, headers=self.headers)
                        if response.status_code != 200:
                            continue
                        result_data = response.json()

                        cast_ok = True
                        director_ok = True

                        if cast_id:
                            cast_ok = PostValidator.has_all_cast(result_data, [cast_id])
                            print(f"🎭 Cast match for {movie_id}: {cast_ok}")

                        if director_name:
                            director_ok = PostValidator.has_director(result_data, director_name)
                            print(f"🎬 Director match for {movie_id}: {director_ok}")

                        if cast_ok and director_ok:
                            validated.append(movie)
                        else:
                            print(f"❌ Skipping {movie_id}: cast_ok={cast_ok}, director_ok={director_ok}")
                    except Exception as e:
                        print(f"⚠️ Validation failed for movie_id={movie_id}: {e}")
                break  # Only apply first matching rule

        return validated or movie_results
    
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
            step = self.validator.inject_path_slot_parameters(
                step, 
                resolved_entities=state.resolved_entities,
                extraction_result=state.extraction_result
            )
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
            # 🛡 Sanity check + refresh after injection
            params = step.get("parameters", {})
            if not isinstance(params, dict):
                print(f"🚨 Malformed parameters in step {step_id} → {type(params)}")
                params = {}
            else:
                assert isinstance(params, dict), f"❌ Step {step_id} has non-dict parameters: {type(params)}"

            # 🧠 Replace placeholders in the path using updated params
            for k, v in params.items():
                if f"{{{k}}}" in path:
                    path = path.replace(f"{{{k}}}", str(v))
                    print(f"🧩 Replaced path slot: {{{k}}} → {v}")

            print(f"🛠️ Resolved full path: {path}")

            path = PathRewriter.rewrite(path, state.resolved_entities)
            full_url = f"{self.base_url}{path}"
            print(f"\n⚡ Executing {step_id}: {path}")

            # Sanitize query parameter if structured from LLM
            if isinstance(params.get("query"), dict):
                original = params["query"]
                params["query"] = original.get("name", "")
                print(f"🔧 Flattened structured query param from {original} → '{params['query']}'")

            try:
                print(f"📤 Calling TMDB: {full_url}")
                print(f"📦 Params: {params}")
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

                        if step["endpoint"].startswith("/discover/movie"):
                            filtered_movies = self._run_post_validations(step, json_data, state)
                            if filtered_movies:
                                query_entities = state.extraction_result.get("query_entities", [])
                                # Add initial final_score to each movie
                                for movie in filtered_movies:
                                    movie["final_score"] = 1.0  # base score

                                ranked = EntityAwareReranker.boost_by_entity_mentions(filtered_movies, query_entities)

                                for movie in ranked:
                                    title = movie.get("title") or movie.get("name")
                                    overview = movie.get("overview", "")
                                    summary = f"{title}: {overview}".strip(": ")
                                    state.responses.append(f"📌 {summary}")
                                ExecutionTraceLogger.log_step(step_id, path, "Validated", state.responses[-1] if state.responses else "")
                            else:
                                state.responses.append("⚠️ No valid results matched all required cast/director.")
                                ExecutionTraceLogger.log_step(step_id, path, "Filtered", "No matching results")
                        else:
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
            text = summary if isinstance(summary, str) else json.dumps(summary)
            print(f"└─ Result: {text[:100]}{'...' if len(text) > 100 else ''}")

# Usage inside orchestrator loop:
# After each response:
# will be moved inside try block where 'summaries' is defined

# On failure:
# will be moved inside exception block where 'response' is defined