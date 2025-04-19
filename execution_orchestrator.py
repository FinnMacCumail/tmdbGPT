from nlp_retriever import PostStepUpdater, PathRewriter, ResultExtractor, expand_plan_with_dependencies
import requests
from copy import deepcopy
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
            print(f"\n▶️ Processing step: {step.get('step_id')}")

            # Skip param injection if already relaxed
            if "_relaxed" not in step.get("step_id", ""):
                step = self.validator.inject_path_slot_parameters(
                    step,
                    resolved_entities=state.resolved_entities,
                    extraction_result=state.extraction_result
                )

            step_id = step.get("step_id")
            depth = step_origin_depth.get(step_id, 0)
            if depth > MAX_CHAIN_DEPTH:
                print(f"🔁 Loop suppression: skipping step {step_id} (depth={depth})")
                i += 1
                continue

            # 🛡 Sanity check on parameters
            params = step.get("parameters", {})
            if not isinstance(params, dict):
                print(f"🚨 Malformed parameters in step {step_id} → {type(params)}")
                params = {}
            else:
                assert isinstance(params, dict), f"❌ Step {step_id} has non-dict parameters: {type(params)}"

            # 🧠 Replace placeholders in the path using updated params
            path = step.get("endpoint")
            for k, v in params.items():
                if f"{{{k}}}" in path:
                    path = path.replace(f"{{{k}}}", str(v))
                    print(f"🧩 Replaced path slot: {{{k}}} → {v}")
            print(f"🛠️ Resolved full path: {path}")
            path = PathRewriter.rewrite(path, state.resolved_entities)
            full_url = f"{self.base_url}{path}"
            print(f"\n⚡ Executing {step_id}: {path}")

            # Sanitize structured query parameter
            if isinstance(params.get("query"), dict):
                original = params["query"]
                params["query"] = original.get("name", "")
                print(f"🔧 Flattened structured query param from {original} → '{params['query']}'")

            # ✅ Deduplication AFTER path + param injection
            param_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            dedup_key = f"{step['endpoint']}?{param_string}"
            step_hash = sha256(dedup_key.encode()).hexdigest()

            if step_hash in seen_step_keys:
                print(f"🔁 Skipping duplicate step_id {step_id} (hash={step_hash}) with same parameters")
                i += 1
                continue

            seen_step_keys.add(step_hash)

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

                        # 🧠 Handle step-specific logic
                        if step["endpoint"].startswith("/discover/movie"):
                            self._handle_discover_movie_step(step, step_id, path, json_data, state, depth, seen_step_keys)
                        else:
                            self._handle_generic_response(step, step_id, path, json_data, state)

                        # Append new steps if needed
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

            # ✅ Always move to the next step after processing
            i += 1

        return state
    
    def _handle_discover_movie_step(self, step, step_id, path, json_data, state, depth=0, seen_step_keys=None):


        seen_step_keys = seen_step_keys or set()

        filtered_movies = self._run_post_validations(step, json_data, state)

        if filtered_movies:
            query_entities = state.extraction_result.get("query_entities", [])
            for movie in filtered_movies:
                movie["final_score"] = 1.0
            ranked = EntityAwareReranker.boost_by_entity_mentions(filtered_movies, query_entities)
            for movie in ranked:
                title = movie.get("title") or movie.get("name")
                overview = movie.get("overview", "")
                summary = f"{title}: {overview}".strip(": ")
                state.responses.append(f"📌 {summary}")
            ExecutionTraceLogger.log_step(step_id, path, "Validated", state.responses[-1])
            return

        # Recovery mode — post-validation failed
        ExecutionTraceLogger.log_step(step_id, path, "Filtered", "No matching results")
        state.responses.append("⚠️ No valid results matched all required cast/director.")

        drop_candidates = ["with_people", "vote_average.gte", "with_genres", "primary_release_year"]
        # Copy current step’s actual parameter state
        current_params = step.get("parameters", {}).copy()
        # Identify all previously dropped filters by looking for _relaxed_ segments
        already_dropped = set()
        if "_relaxed_" in step_id:
            # Split at each '_relaxed_' to extract all dropped parameters
            parts = step_id.split("_relaxed_")[1:]
            already_dropped.update(p.strip() for p in parts if p)
            print(f"Already dropped: {already_dropped}")
            print(f"Remaining: {[p for p in drop_candidates if p not in already_dropped and p in current_params]}")

        # Drop the next filter that hasn't already been dropped
        for param in drop_candidates:
            if param not in current_params or param in already_dropped:
                continue

            retry_step = deepcopy(step)
            retry_step["parameters"] = current_params.copy()
            del retry_step["parameters"][param]

            # Chain multiple drops into step_id
            retry_step["step_id"] = f"{step_id}_relaxed_{param}"

            # Dedup by parameters
            param_string = "&".join(f"{k}={v}" for k, v in sorted(retry_step["parameters"].items()))
            dedup_key = f"{retry_step['endpoint']}?{param_string}"
            retry_hash = sha256(dedup_key.encode()).hexdigest()

            if retry_hash in seen_step_keys:
                continue

            seen_step_keys.add(retry_hash)
            state.plan_steps.insert(0, retry_step)
            ExecutionTraceLogger.log_step(
                step_id, path, "Recovery",
                f"Retrying with {param} dropped → {retry_step['parameters']}"
            )
            print(f"♻️ Retrying by dropping: {param}")
            return  # Only one retry at a time
        # If no retry was possible (all filters dropped or all variants deduped)
        print("🛑 All filter drop retries exhausted.")

        # ✅ Inject trending fallback as last resort
        state.plan_steps.insert(0, {
            "step_id": "fallback_trending",
            "endpoint": "/trending/movie/day",
            "parameters": {},
        })
        ExecutionTraceLogger.log_step(
            step_id, path, "Fallback",
            "Injected trending fallback: /trending/movie/day"
        )
        print("🧭 Injected trending fallback step.")
        return
                
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