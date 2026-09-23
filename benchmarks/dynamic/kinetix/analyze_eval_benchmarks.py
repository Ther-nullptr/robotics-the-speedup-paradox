"""Audit and compare native evaluation benchmarks across a fixed task set."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_eval import comparison_summary  # noqa: E402
from robotics_bench.kinetix.protocol import LEVELS  # noqa: E402


def analyze(reports, *, levels, allow_partial=False, run_completed=True):
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("Expected levels must be nonempty and unique")
    by_task, identity = {}, None
    for report in reports:
        task = report["level"]
        if task in by_task:
            raise ValueError(f"Duplicate task: {task}")
        if task not in levels:
            raise ValueError(f"Unexpected task: {task}")
        if report["status"] != "completed" or not report["compare_preprocess"]:
            raise ValueError(f"Task has no completed paired benchmark: {task}")
        provenance = report["identity"]
        source = {
            key: value
            for key, value in provenance["source_sha256"].items()
            if "/levels/" not in key
        }
        gpu = next(csv.reader([provenance["gpu_query"]], skipinitialspace=True))
        if not source or not provenance["checkpoint"]["sha256"] or len(gpu) != 5:
            raise ValueError(f"Incomplete benchmark identity: {task}")
        current = (
            source,
            provenance["python"],
            provenance["packages"],
            gpu[2:],
            report["flow_steps"],
            report["seeds"],
            report["repeats"],
            report["scope"],
            report["xla_flags"],
            report["policy_metadata"]["parameter_dtypes"],
        )
        if identity is not None and current != identity:
            raise ValueError(
                "Tasks used different code, runtime, hardware or benchmark settings"
            )
        identity = current
        for row in report["rows"]:
            if not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] <= 0:
                raise ValueError(f"Invalid timing sample: {task}")
        # Recompute both gates from raw samples/traces, never trust a stored speedup.
        checks = comparison_summary(report)
        durations = {
            mode: sum(
                row["wall_seconds"] for row in report["rows"] if row["mode"] == mode
            )
            for mode in ("reference", "preprocess_jit")
        }
        reference_traces = report["equivalence_traces"]["reference"]
        repeat_ratios = checks["paired_repeat_speedups"]
        by_task[task] = {
            "task": task,
            "flow_steps": report["flow_steps"],
            "unique_seeds": len(report["seeds"]),
            "repeats": report["repeats"],
            "timed_episode_pairs": len(report["seeds"]) * report["repeats"],
            "reference_seconds": durations["reference"],
            "candidate_seconds": durations["preprocess_jit"],
            "validated_speedup": checks["validated_speedup"],
            "repeat_speedup_min": min(repeat_ratios) if repeat_ratios else None,
            "repeat_speedup_max": max(repeat_ratios) if repeat_ratios else None,
            "trace_equivalent": all(checks["exact_equivalence_by_seed"].values()),
            "timed_outcomes_equal": all(
                checks["timed_outcomes_equal_by_pair"].values()
            ),
            "trace_control_states_per_mode": sum(
                len(t["state_hashes"]) for t in reference_traces
            ),
            "trace_model_outputs_per_mode": sum(
                len(t["action_hashes"]) for t in reference_traces
            ),
            "reference_successes": sum(
                t["result"]["success"] for t in reference_traces
            ),
            "reference_environment_seconds": sum(
                r["components"]["environment_step_seconds"]
                for r in report["rows"]
                if r["mode"] == "reference"
            ),
            "reference_policy_seconds": sum(
                r["components"]["policy_call_seconds"]
                for r in report["rows"]
                if r["mode"] == "reference"
            ),
            "gpu_identity": provenance["gpu_query"],
            "checkpoint_sha256": provenance["checkpoint"]["sha256"],
        }
    complete = set(by_task) == set(levels) and run_completed
    if not complete and not allow_partial:
        raise ValueError(f"Incomplete suite: {len(by_task)}/{len(levels)} tasks")
    tasks = [by_task[task] for task in levels if task in by_task]
    equivalent = bool(tasks) and all(r["validated_speedup"] is not None for r in tasks)
    aggregate = None
    if complete and equivalent:
        baseline = sum(r["reference_seconds"] for r in tasks)
        candidate = sum(r["candidate_seconds"] for r in tasks)
        aggregate = {
            "reference_seconds": baseline,
            "candidate_seconds": candidate,
            "total_rollout_speedup": baseline / candidate,
            "equal_task_geometric_mean_speedup": math.exp(
                sum(math.log(r["validated_speedup"]) for r in tasks) / len(tasks)
            ),
            "timed_episode_pairs": sum(r["timed_episode_pairs"] for r in tasks),
            "trace_control_states_per_mode": sum(
                r["trace_control_states_per_mode"] for r in tasks
            ),
            "trace_model_outputs_per_mode": sum(
                r["trace_model_outputs_per_mode"] for r in tasks
            ),
            "reference_policy_fraction": sum(
                r["reference_policy_seconds"] for r in tasks
            )
            / baseline,
            "reference_environment_fraction": sum(
                r["reference_environment_seconds"] for r in tasks
            )
            / baseline,
        }
    return {
        "format": "kinetix-eval-suite-analysis-v1",
        "levels": levels,
        "complete": complete,
        "all_equivalent": equivalent,
        "tasks": tasks,
        "aggregate": aggregate,
        "timing_scope": "Warm batch-1 episode wall time including reset; excludes file logging, video, traces and cProfile.",
        "aggregation_scope": "Ratio of summed matched rollout durations, not parallel-suite makespan or model inference speedup.",
        "repeat_range": "Observed min/max across timing repeats; not a confidence interval.",
    }


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = report["tasks"]
    figure, ax = plt.subplots(figsize=(9, max(3, len(tasks) * 0.4 + 1.5)))
    ax.axvline(1, color="gray", linestyle="--", linewidth=1)
    for index, row in enumerate(tasks):
        speed = row["validated_speedup"]
        if speed is None:
            ax.text(1, index, "  Bitwise check failed", color="#b7472a", va="center")
            continue
        ax.plot(
            [row["repeat_speedup_min"], row["repeat_speedup_max"]],
            [index, index],
            color="#276e9c",
        )
        ax.plot(speed, index, "o", color="#276e9c")
        ax.annotate(
            f"{speed:.3f}x",
            (row["repeat_speedup_max"], index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(range(len(tasks)), [row["task"] for row in tasks])
    ax.set_ylim(len(tasks) - 0.5, -0.5)
    ax.set(
        xlabel="Reference / candidate rollout wall time",
        title="Native KINETIX: action preprocessing JIT"
        + (" [PARTIAL]" if not report["complete"] else ""),
    )
    ax.margins(x=0.2)
    ax.grid(axis="x", alpha=0.2)
    figure.text(
        0.5,
        0.01,
        "Dots: total ratio per task. Lines: timing-repeat range, not a confidence interval.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"evaluation_speedup.{suffix}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--levels", default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    levels = list(LEVELS) if args.levels == "all" else args.levels.split(",")
    files = [args.input / task / "benchmark.json" for task in levels]
    files = [path for path in files if path.exists()]
    suite = args.input / "suite-manifest.json"
    completed = (
        not suite.exists() or json.loads(suite.read_text())["status"] == "completed"
    )
    report = analyze(
        [json.loads(path.read_text()) for path in files],
        levels=levels,
        allow_partial=args.allow_partial,
        run_completed=completed,
    )
    report["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    output = args.output_dir or args.input / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    table = output / "task_metrics.csv"
    if report["tasks"]:
        with table.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["tasks"][0]))
            writer.writeheader()
            writer.writerows(report["tasks"])
    else:
        table.unlink(missing_ok=True)
    lines = [
        "# Native KINETIX evaluation throughput",
        "",
        f"Coverage: {len(report['tasks'])}/{len(levels)} tasks; complete={report['complete']}; all observed tasks equivalent={report['all_equivalent']}.",
        "",
        report["timing_scope"],
        "",
        report["aggregation_scope"],
        "",
        "| Task | Reference (s) | Candidate (s) | Validated speedup | Trace / timed equality |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["tasks"]:
        speed = (
            f"{row['validated_speedup']:.3f}x"
            if row["validated_speedup"] is not None
            else "unvalidated"
        )
        lines.append(
            f"| {row['task']} | {row['reference_seconds']:.3f} | {row['candidate_seconds']:.3f} | {speed} | {row['trace_equivalent']} / {row['timed_outcomes_equal']} |"
        )
    if report["aggregate"]:
        total = report["aggregate"]
        lines.extend(
            [
                "",
                f"Summed rollout speedup: {total['total_rollout_speedup']:.3f}x. Equal-task geometric mean: {total['equal_task_geometric_mean_speedup']:.3f}x.",
                "",
                f"Verified {total['timed_episode_pairs']} timed episode pairs; {total['trace_control_states_per_mode']} control states and {total['trace_model_outputs_per_mode']} model outputs per mode.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Aggregate speedup withheld until all expected tasks are complete and equivalent.",
            ]
        )
    lines.extend(
        [
            "",
            report["repeat_range"],
            "These seeds test execution equivalence and timing; they do not establish full task-set success rates.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    if not args.no_plots and report["tasks"]:
        plot(report, output)
    print(
        f"Audited {len(report['tasks'])}/{len(levels)} tasks. Report: {output / 'report.md'}"
    )


if __name__ == "__main__":
    main()
