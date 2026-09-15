"""Compare latency summaries with a scalar steady-state cycle model; CPU only."""

import argparse
import html
import json
import math
from pathlib import Path


def _required(mapping: dict, key: str, context: str):
    if key not in mapping:
        raise ValueError(f"{context}.{key} is required")
    return mapping[key]


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}; bool is not a number"
        )
    return value


def _positive(value, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(
            f"{name} must be a finite positive number; bool is not a number"
        )
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _product(left, right, name: str) -> float:
    try:
        value = left * right
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    return _positive(value, name)


def _ratio(numerator: float, denominator: float, name: str) -> float:
    return _positive(numerator / denominator, name)


def chunk_timing(inference_time_ms, n_actions, action_time_ms, overlap_actions) -> dict:
    """Estimate a scalar steady-state cycle; this does not measure wall time.

    Inference and action times must be finite positive scalars. Overlap is an
    integer count of primitive actions from zero through ``n_actions``. A zero
    residual is valid when the overlap window covers all inference time.
    """
    inference_ms = _positive(inference_time_ms, "inference_time_ms")
    n_actions = _integer(n_actions, "n_actions", 1)
    action_time_ms = _positive(action_time_ms, "action_time_ms")
    overlap_actions = _integer(overlap_actions, "overlap_actions")
    if overlap_actions > n_actions:
        raise ValueError("overlap_actions must be <= n_actions")
    action_ms = _product(n_actions, action_time_ms, "action execution time")
    overlap_ms = (
        _product(overlap_actions, action_time_ms, "overlap time")
        if overlap_actions
        else 0.0
    )
    hidden_ms = min(inference_ms, overlap_ms)
    residual_ms = inference_ms - hidden_ms
    # Equivalent to I+n*A-min(I,nprime*A), avoiding cancellation of n*A.
    cycle_ms = _positive(action_ms + residual_ms, "cycle")
    return {
        "hidden_inference_time_ms": hidden_ms,
        "residual_inference_time_ms": residual_ms,
        "cycle_time_ms": cycle_ms,
    }


def _validate(payload) -> tuple[dict, list[dict], dict]:
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    if payload.get("format_version") != "1.0":
        raise ValueError("format_version must be '1.0'")
    if type(payload.get("synthetic")) is not bool:
        raise ValueError("synthetic must be a boolean")
    common = {"format_version": "1.0", "synthetic": payload["synthetic"]}
    for field in ("case_id", "timing_scope", "clock_domain", "baseline_id"):
        common[field] = _text(_required(payload, field, "case"), field)
    common["n_actions"] = _integer(
        _required(payload, "n_actions", "case"), "n_actions", 1
    )
    common["action_time_ms"] = _positive(
        _required(payload, "action_time_ms", "case"), "action_time_ms"
    )
    source_runs = _required(payload, "runs", "case")
    if not isinstance(source_runs, list) or not source_runs:
        raise ValueError("runs must be a nonempty list")
    runs = []
    seen = set()
    for index, source in enumerate(source_runs):
        context = f"runs[{index}]"
        if not isinstance(source, dict):
            raise ValueError(f"{context} must be an object")
        run_id = _text(_required(source, "id", context), f"{context}.id")
        if run_id in seen:
            raise ValueError(f"duplicate run id: {run_id!r}")
        seen.add(run_id)
        run = {"id": run_id}
        for field in ("overlap_actions", "trials", "successes"):
            run[field] = _integer(
                _required(source, field, context),
                f"{context}.{field}",
                1 if field == "trials" else 0,
            )
        if run["overlap_actions"] > common["n_actions"]:
            raise ValueError(f"{context}.overlap_actions must be <= n_actions")
        if run["successes"] > run["trials"]:
            raise ValueError(f"{context}.successes must be <= trials")
        run["inference_time_ms"] = _positive(
            _required(source, "inference_time_ms", context),
            f"{context}.inference_time_ms",
        )
        for field in ("successful_chunk_count_mean", "successful_task_time_mean_ms"):
            value = source.get(field)
            if run["successes"] == 0 and value is not None:
                raise ValueError(f"{context}.{field} must be null with zero successes")
            run[field] = (
                None if value is None else _positive(value, f"{context}.{field}")
            )
        runs.append(run)
    baseline = next((run for run in runs if run["id"] == common["baseline_id"]), None)
    if baseline is None:
        raise ValueError("baseline_id must identify an existing run")
    if baseline["overlap_actions"] != 0:
        raise ValueError("baseline overlap_actions must be 0")
    return common, runs, baseline


def _missing_stat(run: dict, field: str, *, baseline: bool = False) -> str | None:
    prefix = "baseline_" if baseline else ""
    if run["successes"] == 0:
        return f"{prefix}no_successful_trials"
    if run[field] is None:
        return f"{prefix}{field}_not_provided"
    return None


def compare_speedups(payload: dict) -> dict:
    """Compute ratios for one user-declared comparable case, without filling gaps."""
    common, runs, baseline = _validate(payload)
    baseline_inference = baseline["inference_time_ms"]
    baseline_cycle = chunk_timing(
        baseline_inference, common["n_actions"], common["action_time_ms"], 0
    )["cycle_time_ms"]
    baseline_sr = baseline["successes"] / baseline["trials"]
    chunk_field = "successful_chunk_count_mean"
    task_field = "successful_task_time_mean_ms"
    baseline_missing_chunks = _missing_stat(baseline, chunk_field, baseline=True)
    baseline_missing_task = _missing_stat(baseline, task_field, baseline=True)
    baseline_task_estimate = (
        None
        if baseline_missing_chunks
        else _product(baseline[chunk_field], baseline_cycle, "baseline task estimate")
    )
    rows = []
    for run in runs:
        inference_ms = run["inference_time_ms"]
        timing = chunk_timing(
            inference_ms,
            common["n_actions"],
            common["action_time_ms"],
            run["overlap_actions"],
        )
        cycle_ms = timing["cycle_time_ms"]
        missing_chunks = _missing_stat(run, chunk_field)
        missing_task = _missing_stat(run, task_field)
        task_estimate = (
            None
            if missing_chunks
            else _product(run[chunk_field], cycle_ms, f"{run['id']} task estimate")
        )
        missing_reasons = {}
        if missing_chunks:
            missing_reasons["paper_steady_state.successful_task_time_mean_ms"] = (
                missing_chunks
            )
        estimated_reason = missing_chunks or baseline_missing_chunks
        provided_reason = missing_task or baseline_missing_task
        if estimated_reason:
            missing_reasons["speedup_task_estimated"] = estimated_reason
        if provided_reason:
            missing_reasons["speedup_task_success"] = provided_reason
        sr = run["successes"] / run["trials"]
        rows.append(
            {
                "id": run["id"],
                "overlap_actions": run["overlap_actions"],
                "trials": run["trials"],
                "successes": run["successes"],
                "success_rate": sr,
                "success_rate_delta_pp": 100 * (sr - baseline_sr),
                "latency_ratio_alpha": _ratio(
                    inference_ms, baseline_inference, "latency_ratio_alpha"
                ),
                "provided_summary": {
                    "inference_time_ms": inference_ms,
                    chunk_field: run[chunk_field],
                    task_field: run[task_field],
                },
                "paper_steady_state": {
                    **timing,
                    "successful_task_time_mean_ms": task_estimate,
                },
                "speedup_inference": _ratio(
                    baseline_inference, inference_ms, "speedup_inference"
                ),
                "speedup_chunk_estimated": _ratio(
                    baseline_cycle, cycle_ms, "speedup_chunk_estimated"
                ),
                "speedup_task_estimated": (
                    None
                    if estimated_reason
                    else _ratio(
                        baseline_task_estimate, task_estimate, "speedup_task_estimated"
                    )
                ),
                "speedup_task_success": (
                    None
                    if provided_reason
                    else _ratio(
                        baseline[task_field], run[task_field], "speedup_task_success"
                    )
                ),
                "missing_reasons": missing_reasons,
            }
        )
    return {
        **common,
        "estimate_model": "paper_steady_state",
        "comparison_assumption": "user_declared_same_scenario_hardware_statistics_scope",
        "interpretation": {
            "ratios": "baseline_over_run; greater_than_one_is_faster",
            "cycles": "representative_scalar_model; not_measured_cycle_latency",
            "task_success": "ratio_of_provided_success_conditional_means",
            "comparability": "declared_by_input_author; not_verified_by_tool",
        },
        "runs": rows,
    }


def _cell(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return (
        html.escape(str(value))
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def as_markdown(result: dict) -> str:
    ratios = [
        "speedup_inference",
        "speedup_chunk_estimated",
        "speedup_task_estimated",
        "speedup_task_success",
    ]
    columns = ["id", "SR", "delta_SR_pp", "cycle_estimated_ms", "task_estimated_ms"]
    columns += ratios
    lines = [
        f"case_id={_cell(result['case_id'])}; synthetic={str(result['synthetic']).lower()}",
        f"baseline_id={_cell(result['baseline_id'])}; estimate_model=paper_steady_state",
        f"timing_scope={_cell(result['timing_scope'])}; clock_domain={_cell(result['clock_domain'])}; "
        f"n_actions={result['n_actions']}; action_time_ms={_cell(result['action_time_ms'])}",
        "",
        "Ratios > 1 mean faster. Estimates are not measured cycles; null means unavailable.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in result["runs"]:
        values = [
            row["id"],
            f"{100 * row['success_rate']:.2f}%",
            row["success_rate_delta_pp"],
            row["paper_steady_state"]["cycle_time_ms"],
            row["paper_steady_state"]["successful_task_time_mean_ms"],
            *(row[field] for field in ratios),
        ]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    reasons = [
        f"- {_cell(row['id'])}: {_cell(field)} = {_cell(reason)}"
        for row in result["runs"]
        for field, reason in row["missing_reasons"].items()
    ]
    if reasons:
        lines.extend(["", "Missing reasons:", "", *reasons])
    return "\n".join(lines)


def _reject_constant(value: str):
    raise ValueError(f"JSON numbers must be finite: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Independent speedup case JSON"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--markdown", action="store_true", help="Render a summary table")
    mode.add_argument(
        "--json", dest="markdown", action="store_false", help="Emit JSON (default)"
    )
    parser.set_defaults(markdown=False)
    args = parser.parse_args()
    try:
        payload = json.loads(
            args.input.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
        result = compare_speedups(payload)
        output = (
            as_markdown(result)
            if args.markdown
            else json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        )
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(output)


if __name__ == "__main__":
    main()
