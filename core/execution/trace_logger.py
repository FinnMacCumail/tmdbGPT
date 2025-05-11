import json


class ExecutionTraceLogger:
    @staticmethod
    def log_step(step_id, path, status, summary=None, state=None):
        print("\n📍 Execution Trace")
        print(f"├─ Step: {step_id}")
        print(f"├─ Endpoint: {path}")
        print(f"├─ Status: {status}")

        print(
            f"🧾 Trace Entry Added → Step: {step_id}, Status: {status}, Notes: {summary}")

        if summary is not None:
            try:
                text = summary if isinstance(
                    summary, str) else json.dumps(summary, default=str)
            except Exception:
                text = str(summary)
            print(f"└─ Result: {text[:100]}{'...' if len(text) > 100 else ''}")

        if state is not None and hasattr(state, "execution_trace"):
            trace_entry = {
                "step_id": step_id,
                "endpoint": path,
                "status": status,
                "notes": summary if isinstance(summary, str) else str(summary),
                "constraint_tree": str(getattr(state, "constraint_tree", "")),
                "relaxation_log": list(getattr(state, "relaxation_log", [])),
                "injected_steps": [
                    getattr(s, "endpoint", str(s))
                    for s in getattr(state, "steps", [])
                ] if hasattr(state, "steps") else []
            }
            state.execution_trace.append(trace_entry)
