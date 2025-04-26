from nlp_retriever import PostStepUpdater, PathRewriter, ResultExtractor, expand_plan_with_dependencies
import requests
from copy import deepcopy
from hashlib import sha256
from post_validator import PostValidator
from entity_reranker import EntityAwareReranker 
from plan_validator import PlanValidator
import json
from response_formatter import RESPONSE_RENDERERS, format_summary

class ExecutionOrchestrator:
    
    VALIDATION_REGISTRY = [
    {
        "endpoint": "/discover/movie",
        "trigger_param": "with_people",
        "followup_endpoint_template": "/movie/{movie_id}/credits",
        "validator": PostValidator.has_all_cast,
        "args_builder": lambda step, state: {
            "required_ids": [
                int(p) for p in step["parameters"].get("with_people", "").split(",") if p.isdigit()
            ]
        },
        "arg_source": "credits"
    },
    {
        "endpoint": "/discover/movie",
        "trigger_param": "with_people",
        "followup_endpoint_template": "/movie/{movie_id}/credits",
        "validator": PostValidator.has_director,
        "args_builder": lambda step, state: {
            "director_name": next((
                e["name"] for e in state.extraction_result.get("query_entities", [])
                if e.get("type") == "person" and e.get("role") == "director"
            ), None)
        },
        "arg_source": "credits"
    }
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

                query_entities = state.extraction_result.get("query_entities", [])

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

                        # pahase 5.1 - pgpv - 🧩 Dynamically accumulate scores
                        score = 0.0
                        points_per_role = 0.5  # You can tune this later if needed

                        # 🧩 1. Validate Roles Dynamically
                        role_results = PostValidator.validate_roles(result_data, query_entities)

                        for role_key, passed in role_results.items():
                            if passed:
                                score += points_per_role

                        # 🧩 2. Validate Runtime, Year, Rating Filters
                        movie_runtime = movie.get("runtime")
                        release_date = movie.get("release_date") or movie.get("first_air_date")
                        vote_average = movie.get("vote_average", 0)

                        # Check runtime
                        min_runtime = step["parameters"].get("with_runtime.gte")
                        max_runtime = step["parameters"].get("with_runtime.lte")
                        if min_runtime or max_runtime:
                            if PostValidator.meets_runtime(movie, min_minutes=min_runtime, max_minutes=max_runtime):
                                score += 0.3
                                print(f"✅ Runtime OK for {movie_id}")
                            else:
                                print(f"❌ Runtime check failed for {movie_id}")

                        # Check release year
                        target_year = step["parameters"].get("primary_release_year")
                        if target_year and release_date:
                            if str(target_year) in release_date:
                                score += 0.3
                                print(f"✅ Year OK for {movie_id}")
                            else:
                                print(f"❌ Year mismatch for {movie_id}")

                        # Check rating
                        min_rating = step["parameters"].get("vote_average.gte")
                        if min_rating and vote_average >= float(min_rating):
                            score += 0.3
                            print(f"✅ Rating OK for {movie_id}")
                        elif min_rating:
                            print(f"❌ Rating below threshold for {movie_id}")

                        # 🧩 3. Final scoring decision
                        if score > 0:
                            score = min(score, 1.0)  # 🔥 Normalize: max out at 1.0
                            movie["final_score"] = score
                            validated.append(movie)
                            print(f"✅ Movie {movie_id} accepted with final score {score}")
                        else:
                            print(f"❌ Movie {movie_id} rejected (no validations passed)")
                    except Exception as e:
                        print(f"⚠️ Validation failed for movie_id={movie_id}: {e}")
                break  # Only apply first matching rule

        return validated or movie_results
    
    def execute(self, state):
        print(f"\n[DEBUG] Entering Orchestrator Execution")
        print(f"🧭 [DEBUG] Initial question_type: {state.question_type}")
        print(f"🎨 [DEBUG] Initial response_format: {state.response_format}")

        state.error = None
        state.data_registry = {}
        state.completed_steps = []
        seen_step_keys = set()
        step_origin_depth = {}
        MAX_CHAIN_DEPTH = 3

        print(f"🧭 Question Type: {getattr(state, 'question_type', None)}")
        print(f"🎨 Response Format: {getattr(state, 'response_format', None)}")

        while state.plan_steps:
            step = state.plan_steps.pop(0)  # process from front
            step_id = step.get("step_id")

            # 🧩 pase 4 pgpv - NEW: Check if required entities are missing
            missing_requires = [
                req for req in step.get("requires", [])
                if req not in state.resolved_entities
            ]
            if missing_requires:
                print(f"⏭️ Skipping step {step_id}: missing required entities {missing_requires}")
                continue  # Skip step safely

            print(f"\n[DEBUG] Executing Step: {step_id}")
            print(f"[DEBUG] Current question_type: {state.question_type}")
            print(f"[DEBUG] Current response_format: {state.response_format}")

            print(f"▶️ Popped step: {step_id}")
            print(f"🧾 Queue snapshot (after pop): {[s['step_id'] for s in state.plan_steps]}")
            if not state.plan_steps:
                from dependency_manager import DependencyManager
                state = DependencyManager.analyze_dependencies(state)
            if step_id in state.completed_steps:
                print(f"✅ Skipping already completed step: {step_id}")
                continue
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
                    # Fix: handle list injection
                    value = v[0] if isinstance(v, list) else v
                    path = path.replace(f"{{{k}}}", str(value))
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

        # 👇 Safely determine the format type from state
        #format_type = getattr(state, "response_format", "summary")
        format_type = state.response_format or "summary"
        renderer = RESPONSE_RENDERERS.get(format_type, format_summary)

        # 👇 Generate final formatted output
        final_output = renderer(state)

        print("\n--- FINAL RESPONSE ---")
        print(final_output)

        # 👇 You can optionally assign it to state if needed
        state.formatted_response = final_output
        print("[DEBUG] Orchestrator execution completed.")
        return state
    
    def _handle_discover_movie_step(self, step, step_id, path, json_data, state, depth=0, seen_step_keys=None):
        seen_step_keys = seen_step_keys or set()
        print(f"🔎 BEGIN _handle_discover_movie_step for {step_id}")

        filtered_movies = self._run_post_validations(step, json_data, state)

        if filtered_movies:
            print(f"✅ Found {len(filtered_movies)} filtered result(s)")
            query_entities = state.extraction_result.get("query_entities", [])
            for movie in filtered_movies:
                movie["final_score"] = movie.get("final_score", 1.0)
            ranked = EntityAwareReranker.boost_by_entity_mentions(filtered_movies, query_entities)
            state.data_registry[step_id]["validated"] = ranked
            for movie in ranked:
                title = movie.get("title") or movie.get("name")
                overview = movie.get("overview", "")
                summary = f"{title}: {overview}".strip(": ")
                state.responses.append(f"📌 {summary}")
            ExecutionTraceLogger.log_step(step_id, path, "Validated", state.responses[-1])
            state.completed_steps.append(step_id)
            print(f"✅ Step marked completed: {step_id}")
            return

        # Recovery mode — post-validation failed
        ExecutionTraceLogger.log_step(step_id, path, "Filtered", "No matching results")
        state.responses.append("⚠️ No valid results matched all required cast/director.")

        drop_candidates = ["with_people", "vote_average.gte", "with_genres", "primary_release_year"]
        current_params = step.get("parameters", {}).copy()

        already_dropped = set()
        if "_relaxed_" in step_id:
            parts = step_id.split("_relaxed_")[1:]
            already_dropped.update(p.strip() for p in parts if p)
        remaining = [p for p in drop_candidates if p not in already_dropped and p in current_params]
        print(f"🧩 Already dropped: {already_dropped}")
        print(f"🔎 Remaining drop candidates: {remaining}")

        for param in drop_candidates:
            if param not in current_params or param in already_dropped:
                continue

            retry_step = deepcopy(step)
            retry_step["parameters"] = current_params.copy()
            del retry_step["parameters"][param]

            base_id = step_id.split("_relaxed_")[0]
            new_drops = already_dropped.union({param})
            new_suffix = "_relaxed_" + "_relaxed_".join(sorted(new_drops))
            retry_step["step_id"] = f"{base_id}{new_suffix}"

            param_string = "&".join(f"{k}={v}" for k, v in sorted(retry_step["parameters"].items()))
            dedup_key = f"{retry_step['endpoint']}?{param_string}"
            retry_hash = sha256(dedup_key.encode()).hexdigest()

            print(f"🔁 Attempting retry step_id: {retry_step['step_id']}")
            print(f"🔑 Retry hash: {retry_hash}")

            if retry_hash in seen_step_keys:
                print(f"⛔ Duplicate retry hash detected — skipping: {retry_step['step_id']}")
                continue

            if retry_step["step_id"] in state.completed_steps:
                print(f"⛔ Already completed step_id: {retry_step['step_id']}")
                continue

            seen_step_keys.add(retry_hash)
            state.plan_steps.insert(0, retry_step)
            print(f"📥 Plan queue after retry insert: {[s['step_id'] for s in state.plan_steps]}")

            ExecutionTraceLogger.log_step(
                step_id, path, "Recovery",
                f"Retrying with {param} dropped → {retry_step['parameters']}"
            )
            print(f"♻️ Retrying by dropping: {param}")

            # ✅ Prevent reprocessing
            state.completed_steps.append(step_id)
            print(f"✅ Marked original step completed after injecting retry: {step_id}")

            return  # ✅ exit so the new step is handled


        # All retries exhausted
        print("🛑 All filter drop retries exhausted.")

        state.completed_steps.append(step_id)
        print(f"✅ Marked as completed: {step_id}")

        fallback_step = {
            "step_id": "fallback_trending",
            "endpoint": "/trending/movie/day",
            "parameters": {},
        }

        if fallback_step["step_id"] not in state.completed_steps:
            state.plan_steps.insert(0, fallback_step)
            ExecutionTraceLogger.log_step(
                step_id, path, "Fallback",
                "Injected trending fallback: /trending/movie/day"
            )
            print("🧭 Injected trending fallback step.")
        else:
            print("⚠️ Fallback already in completed steps — skipping reinjection.")

        return
    
    def _handle_generic_response(self, step, step_id, path, json_data, state):
        summaries = ResultExtractor.extract(path, json_data, state.resolved_entities)
        query_entities = state.extraction_result.get("query_entities", [])
        role_tagged = any(e.get("role") for e in query_entities)

        # Always apply fallback tagging first
        if step.get("fallback_injected") and isinstance(json_data, dict) and "results" in json_data:
            print(f"♻️ Tagging fallback-injected results from {step['endpoint']}")
            for movie in json_data["results"]:
                movie["final_score"] = 0.3
                movie["source"] = step["endpoint"] + "_relaxed"

        if "credits" in path:
            if role_tagged:
                print(f"🧪 Validating roles from credits for {step_id}")
                results = PostValidator.validate_person_roles(json_data, query_entities)
                cast_ok = results.get("cast_ok", False)
                director_ok = results.get("director_ok", False)

                if cast_ok or director_ok:
                    print("✅ Role validation passed — generating movie_summary")
                    state.responses.append({
                        "type": "movie_summary",
                        "title": "PLACEHOLDER",
                        "overview": "Directed by ...",  # optional
                        "source": path
                    })
                else:
                    print("❌ Role validation failed — but appending fallback summaries")
                    if summaries:
                        state.responses.extend(summaries)
            else:
                print("⚠️ No role specified — appending all extracted summaries")
                if summaries:
                    state.responses.extend(summaries)
        else:
            if summaries:
                state.responses.extend(summaries)

        ExecutionTraceLogger.log_step(step_id, path, "Handled", summaries[:1] if summaries else [])
        state.completed_steps.append(step_id)
        print(f"✅ Step marked completed: {step_id}")
     
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