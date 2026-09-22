"""Matched pilot summaries; simulator host time and paper-model time stay separate."""

import math
from statistics import mean


def failure_budget_time(row, *, inference_ms, action_steps, action_time_ms):
    """Keep measured work; estimate only unexecuted requests/actions after failure."""
    observed = row["paper_model_task_ms"]
    if row["success"] or row["primitive_steps"] == row["max_primitive_steps"]:
        return observed
    missing_steps = row["max_primitive_steps"] - row["primitive_steps"]
    missing_requests = max(
        0, math.ceil(row["max_primitive_steps"] / action_steps) - row["inference_calls"]
    )
    return observed + missing_steps * action_time_ms + missing_requests * inference_ms


def summarize(runs):
    if not runs or runs[0]["id"] != "original":
        raise ValueError("First run must be the original reference")
    coverage = None
    output = []
    for run in runs:
        rows = run["episodes"]
        keys = [
            (
                r.get("task"),
                r["init_state_id"],
                r.get("env_seed"),
                r.get("sampling_seed"),
            )
            for r in rows
        ]
        if (
            not keys
            or len(set(keys)) != len(keys)
            or (coverage is not None and set(keys) != coverage)
        ):
            raise ValueError("Pilot episode coverage must be unique and identical")
        coverage = set(keys)
        if not math.isfinite(run["inference_ms"]) or run["inference_ms"] <= 0:
            raise ValueError("Inference latency must be finite and positive")
        for row in rows:
            if (
                type(row["success"]) is not bool
                or not 0 <= row["primitive_steps"] <= row["max_primitive_steps"]
            ):
                raise ValueError("Invalid episode success or step budget")
            for key in (
                "paper_model_task_ms",
                "paper_model_failure_budget_ms",
                "host_task_seconds",
            ):
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError("Task durations must be finite and positive")
        successful = [r for r in rows if r["success"]]
        result = {
            "id": run["id"],
            "episodes": len(rows),
            "successes": len(successful),
            "success_rate": len(successful) / len(rows),
            "inference_ms": run["inference_ms"],
            "mean_control_steps_failure_budget": mean(
                r["primitive_steps"] if r["success"] else r["max_primitive_steps"]
                for r in rows
            ),
            "mean_control_steps_success": mean(r["primitive_steps"] for r in successful)
            if successful
            else None,
            "mean_task_success_paper_model_ms": mean(
                r["paper_model_task_ms"] for r in successful
            )
            if successful
            else None,
            "mean_task_success_host_seconds": mean(
                r["host_task_seconds"] for r in successful
            )
            if successful
            else None,
            "mean_task_failure_budget_paper_model_ms": mean(
                r["paper_model_failure_budget_ms"] for r in rows
            ),
        }
        baseline = output[0] if output else result
        for metric, source in (
            ("speedup_inference", "inference_ms"),
            ("speedup_task_success_paper_model", "mean_task_success_paper_model_ms"),
            ("speedup_task_success_host", "mean_task_success_host_seconds"),
            (
                "speedup_task_failure_budget_paper_model",
                "mean_task_failure_budget_paper_model_ms",
            ),
        ):
            result[metric] = (
                baseline[source] / result[source]
                if baseline[source] is not None and result[source] is not None
                else None
            )
        output.append(result)
    return output
