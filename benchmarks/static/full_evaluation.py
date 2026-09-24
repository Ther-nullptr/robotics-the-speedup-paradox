"""Manually run matched static evaluation matrices using existing case launchers.

CPU-only planning and aggregation. Actual execution is sequential and explicit;
no downloads, environment installation, simulator modifications or GPU profiling.
"""

import argparse
import csv
from contextlib import contextmanager
import fcntl
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotics_bench.simulators.robocasa import TASK_MAX_STEPS  # noqa: E402
from robotics_bench.statistics import summarize_episodes  # noqa: E402

CASES = ("pi05_libero", "cosmos_libero", "cosmos_robocasa")
RECIPES = ("original-bf16", "optimized-bf16", "int8", "int4")
SHARED = (
    "modulation",
    "gated_residual",
    "cuda_graph",
    "vae_norm_fusion",
    "vae_silu_fusion",
    "vae_condition_prefix",
)
INTEGER = (
    "shared_quant",
    "activation_quant_fusion",
    "integer_pack_reuse",
    "modulation_quant",
)
PYTHON_ENV = {
    "pi05_libero": "ROBOTICS_PI05_PYTHON",
    "cosmos_libero": "ROBOTICS_COSMOS_PYTHON",
    "cosmos_robocasa": "ROBOTICS_COSMOS_ROBOCASA_PYTHON",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--gpu", help="One physical GPU index or full UUID; not needed for summary"
    )
    p.add_argument(
        "--python", help="Case Python executable; defaults to its ROBOTICS_*_PYTHON"
    )
    p.add_argument(
        "--suite",
        choices=("libero_object", "libero_spatial", "libero_goal", "libero_10"),
        default="libero_object",
    )
    p.add_argument(
        "--tasks", default="all", help="RoboCasa task names separated by commas, or all"
    )
    p.add_argument(
        "--episodes",
        type=int,
        help="LIBERO total (default 500); RoboCasa per task (default 50)",
    )
    p.add_argument(
        "--recipes", help="Comma-separated Cosmos recipes; original-bf16 is required"
    )
    p.add_argument("--overlap-actions", type=int, default=2)
    p.add_argument("--seed", type=int, help="Policy seed: PI0.5 42, Cosmos 195")
    p.add_argument(
        "--env-seed",
        type=int,
        default=0,
        help="Cosmos environment seed; RoboCasa increments each episode",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="PI0.5 only; Cosmos remains single-environment",
    )
    p.add_argument("--n-action-steps", type=int, help="PI0.5 5; Cosmos 16")
    p.add_argument("--num-inference-steps", type=int, help="PI0.5 10; Cosmos 5")
    p.add_argument("--integer-tactic", type=int, choices=range(8), default=1)
    p.add_argument(
        "--initialization-retries",
        type=int,
        choices=range(17),
        default=0,
        help="RoboCasa only: observation-only retries against original sync initialization",
    )
    p.add_argument("--layout-id", type=int, default=1)
    p.add_argument("--style-id", type=int, default=1)
    p.add_argument(
        "--record-video",
        action="store_true",
        help="Record complete episodes for every cell",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; no resource checks or output files",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Run the first cell's CPU resource checks only",
    )
    mode.add_argument(
        "--summarize-only",
        action="store_true",
        help="Revalidate and summarize saved results without GPU access",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip audited completed cells; retry unfinished cells in new attempts",
    )
    return p


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def csv_names(value):
    names = value.split(",")
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("Selections must be nonempty and unique")
    return names


def build_plan(args):
    pi05 = args.case == "pi05_libero"
    casa = args.case == "cosmos_robocasa"
    if args.initialization_retries and not casa:
        raise ValueError("Initialization retries are supported only for RoboCasa")
    episodes = args.episodes if args.episodes is not None else (50 if casa else 500)
    seed = args.seed if args.seed is not None else (42 if pi05 else 195)
    actions = (
        args.n_action_steps if args.n_action_steps is not None else (5 if pi05 else 16)
    )
    steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else (10 if pi05 else 5)
    )
    if (
        min(episodes, actions, steps, args.batch_size) < 1
        or min(seed, args.env_seed, args.layout_id, args.style_id) < 0
    ):
        raise ValueError("Counts must be positive and seeds/scene IDs nonnegative")
    if not 1 <= args.overlap_actions <= actions or actions > (
        32 if casa else 16 if not pi05 else 50
    ):
        raise ValueError("Invalid overlap or action horizon")
    if args.gpu is None or not re.fullmatch(
        r"[0-9]+|GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu
    ):
        raise ValueError("Select one GPU index or full UUID")
    if not casa and episodes > 500:
        raise ValueError(
            "Supported LIBERO suites have at most 500 distinct task/initial-state pairs"
        )
    if pi05 and args.batch_size > episodes:
        raise ValueError("PI0.5 batch-size must not exceed episodes")
    if pi05 and args.env_seed != 0:
        raise ValueError(
            "PI0.5 uses its existing evaluator seed protocol; --env-seed is Cosmos-only"
        )
    recipes = (
        csv_names(args.recipes)
        if args.recipes
        else list(RECIPES if not pi05 else RECIPES[:1])
    )
    if pi05 and recipes != ["original-bf16"]:
        raise ValueError("PI0.5 supports only original-bf16 in this matrix")
    if set(recipes) - set(RECIPES) or "original-bf16" not in recipes:
        raise ValueError(
            "Choose known recipes and retain original-bf16 as the shared baseline"
        )
    recipes = [recipe for recipe in RECIPES if recipe in recipes]
    if casa:
        tasks = list(TASK_MAX_STEPS) if args.tasks == "all" else csv_names(args.tasks)
        if set(tasks) - set(TASK_MAX_STEPS):
            raise ValueError("Unknown RoboCasa task")
    else:
        if args.tasks != "all":
            raise ValueError(
                "--tasks is for RoboCasa; LIBERO runs all tasks in --suite"
            )
        tasks = [args.suite]
    executable = args.python or os.environ.get(PYTHON_ENV[args.case], sys.executable)
    executable = shutil.which(executable) or str(
        Path(executable).expanduser().absolute()
    )
    cells = []
    for task in tasks:
        for recipe in recipes:
            for schedule in ("sync", "paper_async"):
                command = [
                    executable,
                    "-u",
                    str(ROOT / "benchmarks/static" / args.case / "run.py"),
                ]
                options = {
                    "gpu": args.gpu,
                    "episodes": episodes,
                    "seed": seed,
                    "n-action-steps": actions,
                    "num-inference-steps": steps,
                    "schedule": schedule,
                    "overlap-actions": args.overlap_actions
                    if schedule == "paper_async"
                    else 0,
                    "model-runtime": "owned",
                    "precision": recipe if recipe in ("int8", "int4") else "bf16",
                    "quant-scope": "text" if pi05 else "dit",
                }
                if casa:
                    options.update(
                        task=task,
                        **{"layout-id": args.layout_id, "style-id": args.style_id},
                    )
                else:
                    options["suite"] = task
                if pi05:
                    options.update(
                        {"batch-size": args.batch_size, "quant-ladder": "none"}
                    )
                    command.append("--no-compile-model")
                else:
                    options.update(
                        {
                            "env-seed": args.env_seed,
                            "integer-tactic": args.integer_tactic,
                        }
                    )
                for key, val in options.items():
                    command.extend(["--" + key, str(val)])
                command.append(
                    "--record-video" if args.record_video else "--no-record-video"
                )
                switches = () if recipe == "original-bf16" else SHARED
                if recipe in ("int8", "int4"):
                    switches += INTEGER
                for switch in switches:
                    command.extend(["--enable", switch])
                reference = f"{task}/original-bf16/sync"
                cell_id = f"{task}/{recipe}/{schedule}"
                if casa and cell_id != reference and args.initialization_retries:
                    command.extend(
                        ["--initialization-retries", str(args.initialization_retries)]
                    )
                cells.append(
                    {
                        "id": cell_id,
                        "task": task,
                        "recipe": recipe,
                        "schedule": schedule,
                        "episodes": episodes,
                        "command": command,
                        "reference": reference
                        if casa and cell_id != reference
                        else None,
                    }
                )
    source_files = [Path(__file__), *sorted((ROOT / "src").rglob("*.py"))]
    source_hash = digest(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        }
    )
    return {
        "format": "static-full-evaluation-v1",
        "case": args.case,
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "cells": cells,
        "source_sha256": source_hash,
        "expected_episodes": sum(c["episodes"] for c in cells),
    }


def next_attempt(root, cell):
    index = 1
    while True:
        output = root / cell["id"] / f"attempt-{index:03d}"
        log = output.with_suffix(".launcher.log")
        if not any(p.exists() or p.is_symlink() for p in (output, log)):
            return output
        index += 1


@contextmanager
def matrix_lock(root):
    """Sibling lock also protects initial directory creation and summary writes."""
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.with_name(root.name + ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "This matrix is already running or being summarized"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def save_state(state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(Path(state["plan"]["output_dir"]) / "matrix.json", state)


def open_state(plan, *, resume):
    root = Path(plan["output_dir"])
    if resume:
        state = read_json(root / "matrix.json")
        if state["plan"] != plan:
            raise ValueError(
                "Saved plan differs; restore the original arguments/code or use a new output directory"
            )
        return state
    root.mkdir(parents=True, exist_ok=False)
    state = {"plan": plan, "status": "planned", "cells": {}}
    save_state(state)
    return state


def summary_module():
    spec = importlib.util.spec_from_file_location(
        "_matrix_summary", ROOT / "tools/summarize_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_completed(cell, record):
    run = Path(record["output_dir"])
    manifest = read_json(run / "case-manifest.json")
    if manifest.get("case_fingerprint") != record.get("case_fingerprint"):
        raise ValueError(f"Run identity differs: {cell['id']}")
    helper = summary_module()
    path, rows = helper.read_ledger(run)
    if (
        helper.completion_status(path, rows, False) != "complete"
        or len(rows) != cell["episodes"]
    ):
        raise ValueError(f"Incomplete episode coverage: {cell['id']}")
    budget = read_json(run / "coverage.json")["max_primitive_steps"]
    summarize_episodes(rows, max_steps=budget)
    if cell.get("reference") and any(
        row.get("reference_initialization_passed") is not True for row in rows
    ):
        raise ValueError(
            f"RoboCasa reference initialization is unverified: {cell['id']}"
        )
    return rows, budget


def episode_keys(rows):
    return sorted((r["task"], r["init_state_id"], r.get("env_seed")) for r in rows)


def summarize(state):
    groups, reference_keys, reference_budgets = {}, {}, {}
    completed = 0
    for cell in state["plan"]["cells"]:
        group = groups.setdefault(
            (cell["recipe"], cell["schedule"]),
            {
                "rows": [],
                "budgets": {},
                "complete": True,
                "completed_cells": 0,
                "expected_cells": 0,
            },
        )
        group["expected_cells"] += 1
        record = state["cells"].get(cell["id"], {})
        if record.get("status") != "completed":
            group["complete"] = False
            continue
        rows, budget = load_completed(cell, record)
        keys = episode_keys(rows)
        if reference_budgets.setdefault(cell["task"], budget) != budget:
            raise ValueError(f"Mismatched task budgets: {cell['id']}")
        if cell["task"] in reference_keys and reference_keys[cell["task"]] != keys:
            raise ValueError(f"Mismatched episode sets: {cell['id']}")
        reference_keys[cell["task"]] = keys
        group["rows"].extend(rows)
        for row in rows:
            previous = group["budgets"].setdefault(row["task"], budget)
            if previous != budget:
                raise ValueError("Conflicting task budgets")
        group["completed_cells"] += 1
        completed += 1
    result = []
    for (recipe, schedule), group in groups.items():
        row = {
            "recipe": recipe,
            "schedule": schedule,
            "complete": group["complete"],
            "completed_cells": group["completed_cells"],
            "expected_cells": group["expected_cells"],
            **dict.fromkeys(
                (
                    "episodes",
                    "successes",
                    "success_rate",
                    "failure_budget_total_steps",
                    "failure_budget_mean_steps",
                    "success_total_steps",
                    "success_mean_steps",
                )
            ),
        }
        if group["complete"]:
            stats = summarize_episodes(group["rows"], task_max_steps=group["budgets"])[
                "overall"
            ]
            row.update(
                episodes=stats["episodes"],
                successes=stats["successes"],
                success_rate=stats["success_rate"],
                failure_budget_total_steps=stats["failure_penalized"][
                    "total_control_steps"
                ],
                failure_budget_mean_steps=stats["failure_penalized"][
                    "mean_control_steps"
                ],
                success_total_steps=stats["successful_episodes"]["total_control_steps"],
                success_mean_steps=stats["successful_episodes"]["mean_control_steps"],
            )
        result.append(row)
    return {
        "case": state["plan"]["case"],
        "complete": completed == len(state["plan"]["cells"]),
        "completed_cells": completed,
        "expected_cells": len(state["plan"]["cells"]),
        "aggregation": "episode_weighted",
        "rows": result,
    }


def write_report(state):
    report = summarize(state)
    root = Path(state["plan"]["output_dir"])
    write_json(root / "matrix-summary.json", report)
    with (root / "matrix-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    lines = [
        "# Static evaluation matrix",
        "",
        f"Case: {report['case']}. Completed cells: {report['completed_cells']}/{report['expected_cells']}.",
        "",
        "Missing cells produce no aggregate metrics. Failure penalties use each task's declared budget.",
        "These are control-step/quality results, not measured task or inference speedups.",
        "",
        "| Recipe | Schedule | Cells | Episodes | Successes | Success rate | Budget total steps | Budget mean steps | Success total steps | Success mean steps |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def display(value):
        return (
            "—"
            if value is None
            else f"{value:.6f}"
            if isinstance(value, float)
            else str(value)
        )

    for row in report["rows"]:
        values = [
            row["recipe"],
            row["schedule"],
            f"{row['completed_cells']}/{row['expected_cells']}",
            *[
                row[k]
                for k in (
                    "episodes",
                    "successes",
                    "success_rate",
                    "failure_budget_total_steps",
                    "failure_budget_mean_steps",
                    "success_total_steps",
                    "success_mean_steps",
                )
            ],
        ]
        lines.append("| " + " | ".join(map(display, values)) + " |")
    (root / "matrix-summary.md").write_text("\n".join(lines) + "\n")
    return report


def preflight(plan):
    # A nonexistent path allows read-only preflight even when resuming old outputs.
    output = Path(plan["output_dir"]) / ("preflight-" + uuid.uuid4().hex)
    command = [*plan["cells"][0]["command"], "--output-dir", str(output), "--dry-run"]
    print("CPU preflight: " + shlex.join(command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"CPU preflight failed:\n{result.stdout}\n{result.stderr}")
    identity = json.loads(result.stdout)["identity"]
    # Configuration variants intentionally differ; the resource/source identity must not.
    return {
        k: v
        for k, v in identity.items()
        if k not in ("case", "optimizations", "model_runtime")
    }


def execute(state):
    root = Path(state["plan"]["output_dir"])
    state["status"] = "running"
    save_state(state)
    for cell in state["plan"]["cells"]:
        record = state["cells"].get(cell["id"], {})
        if record.get("status") == "completed":
            load_completed(cell, record)
            print(f"Skip completed: {cell['id']}", flush=True)
            continue
        output = next_attempt(root, cell)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [*cell["command"], "--output-dir", str(output)]
        if cell["reference"]:
            reference = state["cells"][cell["reference"]]
            if reference["status"] != "completed":
                raise ValueError("RoboCasa reference must complete first")
            command.extend(["--reference-run", reference["output_dir"]])
        log = output.with_suffix(".launcher.log")
        state["cells"][cell["id"]] = record = {
            "status": "running",
            "output_dir": str(output),
            "command": command,
            "launcher_log": str(log),
        }
        save_state(state)
        print(f"Run {cell['id']}\n  {shlex.join(command)}\n  Log: {log}", flush=True)
        process = None
        try:
            with log.open("x") as stream:
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    start_new_session=True,
                )
                code = process.wait()
            if code:
                raise RuntimeError(f"Cell exited with code {code}; see {log}")
            manifest = read_json(output / "case-manifest.json")
            context = {
                k: v
                for k, v in manifest["identity"].items()
                if k not in ("case", "optimizations", "model_runtime")
            }
            if context != state["resource_identity"]:
                raise ValueError("Resource/source identity changed during evaluation")
            record["case_fingerprint"] = manifest["case_fingerprint"]
            load_completed(cell, record)
            record["status"] = "completed"
            write_report(state)
        except BaseException as exc:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            state["status"] = "failed"
            save_state(state)
            raise
        save_state(state)
    write_report(state)
    state["status"] = "completed"
    save_state(state)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.summarize_only:
            with matrix_lock(args.output_dir.expanduser().resolve()):
                state = read_json(
                    args.output_dir.expanduser().resolve() / "matrix.json"
                )
                if state["plan"]["case"] != args.case:
                    raise ValueError("Case differs from the saved matrix")
                report = write_report(state)
            print(
                f"Completed cells: {report['completed_cells']}/{report['expected_cells']}"
            )
            return 0 if report["complete"] else 1
        plan = build_plan(args)
        if args.dry_run:
            print(
                f"Case: {plan['case']}; cells: {len(plan['cells'])}; planned episodes: {plan['expected_episodes']}"
            )
            for cell in plan["cells"]:
                command = [
                    *cell["command"],
                    "--output-dir",
                    str(Path(plan["output_dir"]) / cell["id"] / "attempt-001"),
                ]
                if cell["reference"]:
                    command.extend(
                        [
                            "--reference-run",
                            str(
                                Path(plan["output_dir"])
                                / cell["reference"]
                                / "attempt-001"
                            ),
                        ]
                    )
                print(shlex.join(command))
            return 0
        identity = preflight(plan)
        if args.preflight:
            print(
                "CPU resource preflight passed. GPU libraries, task/scene compatibility and model execution remain unchecked."
            )
            return 0
        with matrix_lock(Path(plan["output_dir"])):
            state = open_state(plan, resume=args.resume)
            if state.get("resource_identity", identity) != identity:
                raise ValueError(
                    "Resource/source identity changed; use a new matrix directory"
                )
            state["resource_identity"] = identity
            previous_term = signal.getsignal(signal.SIGTERM)

            def terminate(signum, frame):
                raise KeyboardInterrupt("Termination requested")

            signal.signal(signal.SIGTERM, terminate)
            try:
                execute(state)
            finally:
                signal.signal(signal.SIGTERM, previous_term)
        print(f"Completed. Report: {Path(plan['output_dir']) / 'matrix-summary.md'}")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        p.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
