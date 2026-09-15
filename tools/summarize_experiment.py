"""Summarize existing episode ledgers; CPU only, no model or simulator imports."""

import argparse
import csv
import html
import importlib.util
import io
import json
from pathlib import Path
import sys


def _statistics_module():
    # The runtime loads this utility by file, without trusting an external
    # model checkout's sys.path to resolve the statistics implementation.
    spec = importlib.util.spec_from_file_location(
        "_robotics_episode_statistics",
        Path(__file__).with_name("episode_statistics.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize_episodes = _statistics_module().summarize_episodes


def read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_ledger(source):
    path = source.expanduser().resolve()
    if path.is_dir():
        path /= "episodes.jsonl"
    if not path.is_file():
        raise ValueError(
            f"Episode ledger not found: {path}; provide a run directory or JSONL file"
        )
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected an episode object")
        rows.append(row)
    return path, rows


def completion_status(path, rows, allow_partial):
    manifest_path = path.parent / "case-manifest.json"
    coverage_path = path.parent / "coverage.json"
    manifest = read_object(manifest_path) if manifest_path.exists() else None
    coverage = read_object(coverage_path) if coverage_path.exists() else None
    if manifest is None and coverage is None:
        return "unverified"
    partial = (manifest is not None and manifest.get("status") != "completed") or (
        coverage is not None and coverage.get("status") != "passed"
    )
    if partial:
        if not allow_partial:
            raise ValueError(
                f"{path.parent}: run is unfinished or failed; use --allow-partial to summarize recorded episodes"
            )
        return "partial"
    counts = []
    if manifest is not None:
        expected = manifest.get("identity", {}).get("case", {}).get("episodes")
        if expected is not None:
            counts.append(expected)
    if coverage is not None:
        for key in ("expected_episodes", "completed_episodes"):
            if key in coverage:
                counts.append(coverage[key])
        successes = coverage.get("successes")
        if successes is not None and (
            type(successes) is not int
            or successes != sum(row["success"] for row in rows)
        ):
            raise ValueError(
                f"{coverage_path}: success count disagrees with episode ledger"
            )
        if "tasks" in coverage:
            actual = {}
            for row in rows:
                actual.setdefault(row["task"], []).append(row["init_state_id"])
            if set(actual) != set(coverage["tasks"]):
                raise ValueError(
                    f"{coverage_path}: task coverage disagrees with episode ledger"
                )
            for task, ids in actual.items():
                declared = coverage["tasks"][task]
                if declared.get("episodes") != len(ids) or declared.get(
                    "init_state_ids"
                ) != sorted(ids):
                    raise ValueError(
                        f"{coverage_path}: initial-state coverage disagrees for {task}"
                    )
    if any(type(count) is not int or count != len(rows) for count in counts):
        raise ValueError(
            f"{path.parent}: completed run count disagrees with episode ledger"
        )
    return (
        "complete"
        if manifest is not None and coverage is not None and counts
        else "unverified"
    )


def task_budgets(values):
    budgets = {}
    for value in values:
        try:
            task, limit = value.rsplit("=", 1)
            limit = int(limit)
        except ValueError as exc:
            raise ValueError("--task-max-steps must be TASK=POSITIVE_INTEGER") from exc
        if not task.strip() or limit <= 0 or task in budgets:
            raise ValueError(
                "--task-max-steps requires unique task names and positive budgets"
            )
        budgets[task] = limit
    return budgets


def build_report(args):
    sources = [read_ledger(path) for path in args.input]
    paths = [path for path, _ in sources]
    if len(set(paths)) != len(paths):
        raise ValueError("The same episode ledger was supplied more than once")
    budgets = task_budgets(args.task_max_steps)
    # Each CLI budget must apply somewhere, even when the runs cover different tasks.
    all_tasks = {
        row.get("task")
        for _, rows in sources
        for row in rows
        if isinstance(row.get("task"), str)
    }
    unknown = set(budgets) - all_tasks
    if unknown:
        raise ValueError(f"Unknown tasks in --task-max-steps: {sorted(unknown)}")
    runs = []
    for path, rows in sources:
        names = {row.get("task") for row in rows if isinstance(row.get("task"), str)}
        overrides = {name: value for name, value in budgets.items() if name in names}
        max_steps = args.max_steps
        budget_source = "cli" if max_steps is not None else "not_provided"
        if max_steps is None and (path.parent / "coverage.json").exists():
            max_steps = read_object(path.parent / "coverage.json").get(
                "max_primitive_steps"
            )
            if max_steps is not None:
                budget_source = "coverage.json"
        try:
            statistics = summarize_episodes(
                rows,
                max_steps=max_steps,
                task_max_steps=overrides,
            )
            completion = completion_status(path, rows, args.allow_partial)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        runs.append(
            {
                "name": path.parent.name
                if path.name == "episodes.jsonl"
                else path.stem,
                "input": str(path),
                "completion": completion,
                "budgets": {
                    "max_steps": max_steps,
                    "task_max_steps": overrides,
                    "source": ("mixed" if budget_source == "coverage.json" else "cli")
                    if overrides
                    else budget_source,
                    "global_source": budget_source,
                    "task_override_source": "cli" if overrides else "not_provided",
                },
                "statistics": statistics,
            }
        )
    return {
        "format_version": "episode-statistics-v1",
        "unit": "primitive_control_steps",
        "runs": runs,
    }


def write_run_summary(output_dir, rows, *, max_steps=None):
    """Save an already audited completed run using the same CPU statistics API.

    The caller supplies actual runtime budgets and calls this only after its
    episode/summary checks pass. No manifest or live simulator is read here.
    """
    output_dir = Path(output_dir)
    report = {
        "format_version": "episode-statistics-v1",
        "unit": "primitive_control_steps",
        "runs": [
            {
                "name": output_dir.name,
                "input": str(output_dir / "episodes.jsonl"),
                "completion": "complete",
                "budgets": {
                    "max_steps": max_steps,
                    "task_max_steps": {},
                    "source": "runtime" if max_steps is not None else "not_provided",
                    "global_source": "runtime"
                    if max_steps is not None
                    else "not_provided",
                    "task_override_source": "not_provided",
                },
                "statistics": summarize_episodes(rows, max_steps=max_steps),
            }
        ],
    }
    for filename, content in (
        (
            "episode-summary.json",
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        ),
        ("episode-summary.md", markdown_report(report, per_task=True)),
    ):
        with (output_dir / filename).open("x", encoding="utf-8") as stream:
            stream.write(content)
    return report


def number(value):
    if value is None:
        return "N/A"
    return (
        str(value) if isinstance(value, int) else f"{value:.6f}".rstrip("0").rstrip(".")
    )


def label(value):
    return (
        html.escape(str(value))
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def markdown_report(report, per_task=False):
    lines = [
        "# Experiment control-step summary",
        "",
        "Unit: primitive control steps. Task means are weighted by episode count; success-only means are weighted by successful episode count.",
        "",
    ]
    for run in report["runs"]:
        stats = run["statistics"]["overall"]
        lines += [
            f"## {label(run['name'])}",
            "",
            f"Status: {run['completion']}; episodes: {stats['episodes']}; successes: {stats['successes']}; success rate: {number(stats['success_rate'] * 100)}%.",
            "",
            "| Scope | Episodes | Total control steps | Mean control steps |",
            "| --- | ---: | ---: | ---: |",
        ]
        for title, key, count in [
            ("All episodes (observed)", "all_episodes", stats["episodes"]),
            ("Success only", "successful_episodes", stats["successes"]),
            ("All episodes (failure budget)", "failure_penalized", stats["episodes"]),
        ]:
            item = stats[key]
            lines.append(
                f"| {title} | {count} | {number(item['total_control_steps'])} | {number(item['mean_control_steps'])} |"
            )
        penalty = stats["failure_penalized"]
        if penalty["status"] != "available":
            lines.append("")
            lines.append(
                "Failure-budget statistics unavailable for tasks: "
                + ", ".join(label(name) for name in penalty["missing_budget_tasks"])
                + ". Supply --max-steps or --task-max-steps.",
            )
        if per_task:
            lines += [
                "",
                "| Task | Episodes | Successes | Observed mean | Success mean | Failure-budget mean | Episode weight | Success weight |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for task in run["statistics"]["tasks"]:
                values = [
                    label(task["task"]),
                    task["episodes"],
                    task["successes"],
                    task["all_episodes"]["mean_control_steps"],
                    task["successful_episodes"]["mean_control_steps"],
                    task["failure_penalized"]["mean_control_steps"],
                    task["trial_weight"],
                    task["success_weight"],
                ]
                lines.append(
                    "| "
                    + " | ".join(
                        str(value) if index == 0 else number(value)
                        for index, value in enumerate(values)
                    )
                    + " |"
                )
        lines.append("")
    return "\n".join(lines)


def csv_report(report, per_task=False):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "run",
            "completion",
            "scope",
            "task",
            "episodes",
            "successes",
            "failures",
            "success_rate",
            "all_total_control_steps",
            "all_mean_control_steps",
            "success_total_control_steps",
            "success_mean_control_steps",
            "failure_penalized_total_control_steps",
            "failure_penalized_mean_control_steps",
            "trial_weight",
            "success_weight",
            "penalty_status",
            "missing_budget_tasks",
        ]
    )
    for run in report["runs"]:
        entries = [("overall", run["statistics"]["overall"])]
        if per_task:
            entries.extend(("task", task) for task in run["statistics"]["tasks"])
        for scope, stats in entries:
            penalty = stats["failure_penalized"]
            values = [
                run["name"],
                run["completion"],
                scope,
                stats.get("task", ""),
                stats["episodes"],
                stats["successes"],
                stats["failures"],
                stats["success_rate"],
            ]
            for key in (
                "all_episodes",
                "successful_episodes",
                "failure_penalized",
            ):
                values += [
                    stats[key]["total_control_steps"],
                    stats[key]["mean_control_steps"],
                ]
            values += [
                stats.get("trial_weight"),
                stats.get("success_weight"),
                penalty["status"],
                ";".join(penalty["missing_budget_tasks"]),
            ]
            writer.writerow(values)
    return stream.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run directories or episodes.jsonl files; runs stay separate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Declared control-step budget for failed episodes; never inferred from observed maxima",
    )
    parser.add_argument(
        "--task-max-steps",
        action="append",
        default=[],
        metavar="TASK=STEPS",
        help="Override the global budget for a task (repeatable)",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "csv"),
        help="Default: infer from --output extension, otherwise markdown",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New output file; existing files are not overwritten",
    )
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="Include task rows in Markdown/CSV; JSON always contains task statistics",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly summarize recorded episodes of unfinished/failed runs",
    )
    args = parser.parse_args(argv)
    try:
        report = build_report(args)
        format_name = args.format or (
            {".json": "json", ".csv": "csv"}.get(args.output.suffix.lower(), "markdown")
            if args.output
            else "markdown"
        )
        result = (
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            if format_name == "json"
            else csv_report(report, args.per_task)
            if format_name == "csv"
            else markdown_report(report, args.per_task)
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="") as stream:
                stream.write(result)
            print(args.output)
        else:
            sys.stdout.write(result)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
