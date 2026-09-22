"""Run a matched synchronous Cosmos LIBERO quantization pilot in one process.

This is a small task pilot, not a full-suite accuracy evaluation. Hardware policy
time, actual simulator-host task time and estimated physical task time are kept
separate. No smoothing, rotation, training or automatic resource download occurs.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--suite",
        choices=("libero_object", "libero_spatial", "libero_goal", "libero_10"),
        default="libero_object",
    )
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--initial-states", type=int, nargs="+", default=[0, 1])
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--action-steps", type=int, default=16)
    p.add_argument("--action-time-ms", type=float, default=50)
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument(
        "--tiers",
        type=int,
        choices=range(11),
        nargs="+",
        help="After fixed precision, optionally select appendix tiers 0..10",
    )
    p.add_argument("--integer-tactic", type=int, choices=range(8), default=1)
    p.add_argument("--profile-skill", type=Path)
    p.add_argument("--render-python", default=sys.executable)
    for name in (
        "cosmos-source",
        "checkpoint",
        "dataset-stats",
        "text-embeddings",
        "vae-checkpoint",
        "libero-config-dir",
    ):
        p.add_argument("--" + name, type=Path)
    return p


def resource(args, name, variable):
    value = getattr(args, name) or os.environ.get(variable)
    if not value:
        raise ValueError(f"Provide --{name.replace('_', '-')} or {variable}")
    result = Path(value).expanduser().resolve()
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if (
        args.gpu < 0
        or args.steps < 1
        or not 1 <= args.action_steps <= 16
        or args.repeats < 3
        or not math.isfinite(args.action_time_ms)
        or args.action_time_ms <= 0
    ):
        raise ValueError("Invalid GPU, inference/action settings or repetitions")
    if (
        not args.initial_states
        or len(set(args.initial_states)) != len(args.initial_states)
        or min(args.initial_states) < 0
    ):
        raise ValueError("Initial states must be unique nonnegative IDs")
    if args.tiers is not None and len(set(args.tiers)) != len(args.tiers):
        raise ValueError("Duplicate tiers")
    assets = {
        name: resource(args, name, variable)
        for name, variable in (
            ("cosmos_source", "ROBOTICS_COSMOS_SOURCE"),
            ("checkpoint", "ROBOTICS_COSMOS_CHECKPOINT"),
            ("dataset_stats", "ROBOTICS_COSMOS_DATASET_STATS"),
            ("text_embeddings", "ROBOTICS_COSMOS_TEXT_EMBEDDINGS"),
            ("vae_checkpoint", "ROBOTICS_COSMOS_VAE_CHECKPOINT"),
            ("libero_config_dir", "ROBOTICS_COSMOS_LIBERO_CONFIG_DIR"),
        )
    }
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID=str(args.gpu),
        LIBERO_CONFIG_PATH=str(assets["libero_config_dir"]),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="disabled",
    )
    import numpy as np
    import torch
    from robotics_bench.engines.cosmos import CosmosEngine
    from robotics_bench.optimizations.config import OptimizationConfig
    from robotics_bench.optimizations.cosmos import optimize_cosmos
    from robotics_bench.profiling.task_report import summarize, failure_budget_time
    from robotics_bench.protocols.static_runner import run_episode
    from robotics_bench.simulators.libero import LiberoSuite

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    suite = LiberoSuite(args.suite)
    task = next((t for t in suite.tasks() if t["id"] == args.task_id), None)
    if task is None or max(args.initial_states) >= task["initial_states"]:
        raise ValueError("Task or initial-state selection is unavailable")
    budget = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }[args.suite]
    common = ("modulation", "gated_residual", "cuda_graph")
    integer = (*common, "shared_quant", "activation_quant_fusion", "integer_pack_reuse")
    configurations = [
        ("original", OptimizationConfig(scopes=("dit",))),
        ("optimized-bf16", OptimizationConfig(common, scopes=("dit",))),
    ]
    if args.tiers is None:
        configurations += [
            (p, OptimizationConfig(integer, p, ("dit",), args.integer_tactic))
            for p in ("int8", "int4")
        ]
    else:
        configurations += [
            (
                "w8a8" if t == 0 else f"w4-t{t}",
                OptimizationConfig(integer, "int8", ("dit",), args.integer_tactic, t),
            )
            for t in args.tiers
        ]
    manifest = {
        "format": "cosmos-libero-quantization-pilot-v1",
        "status": "running",
        "suite": args.suite,
        "task": task,
        "initial_states": args.initial_states,
        "sampling_seed": args.seed,
        "env_seed": args.env_seed,
        "steps": args.steps,
        "action_steps": args.action_steps,
        "paper_action_time_ms": args.action_time_ms,
        "max_primitive_steps": budget,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_selector": args.gpu,
        "torch": torch.__version__,
        "assets": {k: str(v) for k, v in assets.items()},
        "time_definition": "Host task time excludes reset/settling and warmup. Paper sync finite task = sum(measured request policy times) + executed steps * declared action time; it is an estimate, not simulator wall time.",
        "configurations": [{"id": name, **c.to_dict()} for name, c in configurations],
    }
    manifest["git_head"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest["git_status"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    )
    manifest["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "src").rglob("*")
        if p.suffix in (".py", ".cu", ".cuh")
    }
    manifest["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    engine = simulator = None
    runs = []
    reference = None
    try:
        engine = CosmosEngine(
            assets["cosmos_source"],
            assets["checkpoint"],
            assets["dataset_stats"],
            assets["text_embeddings"],
            assets["vae_checkpoint"],
            num_inference_steps=args.steps,
        )
        engine.load([task["description"]], output / "checkpoint-load.json")
        engine.configure_optimizations(OptimizationConfig(scopes=("dit",)))
        manifest["engine"] = deepcopy(engine.metadata)
        simulator = suite.make_simulator(args.task_id, env_seed=args.env_seed)

        class ClockedSimulator:
            def reset(self, *a, **k):
                value = simulator.reset(*a, **k)
                self.started = time.perf_counter()
                return value

            def __getattr__(self, name):
                return getattr(simulator, name)

        class ClockedEngine:
            def reset(self, identity):
                self.request_ms = []
                engine.reset(identity)

            def infer_chunk(self, *a):
                start = time.perf_counter()
                value = engine.infer_chunk(*a)
                torch.cuda.synchronize()
                self.request_ms.append((time.perf_counter() - start) * 1000)
                return value

        clocked_sim, clocked_engine = ClockedSimulator(), ClockedEngine()
        for name, configuration in configurations:
            destination = output / name
            destination.mkdir()
            with (
                torch.inference_mode(),
                optimize_cosmos(engine._model, configuration) as coverage,
            ):
                observation = simulator.reset(
                    args.initial_states[0], seed=args.env_seed
                )
                engine.reset((name, "warmup"))
                for _ in range(3):
                    engine.infer_chunk(observation, task["description"], args.seed)
                samples = []
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    action = engine.infer_chunk(
                        observation, task["description"], args.seed
                    )
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
                    if not np.isfinite(action).all():
                        raise ValueError("Nonfinite policy actions")
                if reference is None:
                    reference = action.copy()
                np.save(destination / "probe-actions.npy", action)
                exact = bool(np.array_equal(action, reference))
                error = float(
                    np.max(
                        np.abs(action.astype(np.float64) - reference.astype(np.float64))
                    )
                )
                measured_at = datetime.now(timezone.utc).isoformat()
                rows = []
                with (destination / "episodes.jsonl").open("w") as ledger:
                    for state in args.initial_states:
                        row = run_episode(
                            clocked_engine,
                            clocked_sim,
                            task=task["name"],
                            init_state_id=state,
                            env_seed=args.env_seed,
                            sampling_seed=args.seed,
                            max_steps=budget,
                            n_action_steps=args.action_steps,
                        )
                        row["host_task_seconds"] = (
                            time.perf_counter() - clocked_sim.started
                        )
                        row["sampling_seed"] = args.seed
                        row["inference_ms"] = clocked_engine.request_ms
                        row["paper_model_task_ms"] = (
                            sum(clocked_engine.request_ms)
                            + row["primitive_steps"] * args.action_time_ms
                        )
                        row["paper_model_failure_budget_ms"] = failure_budget_time(
                            row,
                            inference_ms=median(samples),
                            action_steps=args.action_steps,
                            action_time_ms=args.action_time_ms,
                        )
                        rows.append(row)
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                        ledger.flush()
                        print(
                            json.dumps(
                                {
                                    "configuration": name,
                                    "initial_state": state,
                                    "success": row["success"],
                                    "steps": row["primitive_steps"],
                                }
                            ),
                            flush=True,
                        )
                runs.append(
                    {
                        "id": name,
                        "config": configuration.to_dict(),
                        "coverage": deepcopy(coverage),
                        "inference_samples_ms": samples,
                        "inference_ms": median(samples),
                        "episodes": rows,
                        "precision": configuration.precision_label,
                        "exact": exact,
                        "max_abs": error,
                        "recorded_at": measured_at,
                    }
                )
                (output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
                (output / "summary.json").write_text(
                    json.dumps(summarize(runs), indent=2) + "\n"
                )
        manifest["status"] = "completed"
        summary = summarize(runs)
        lines = [
            "# Cosmos LIBERO quantization pilot",
            "",
            "Small matched pilot; not a full-suite success-rate estimate.",
            "",
            "| Configuration | Successes / episodes | Inference ms | Inference speedup | Mean steps (failure budget) | Mean steps (success) | Task speedup (success, paper model) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary:

            def number(value):
                return "N/A" if value is None else f"{value:.4f}"

            lines.append(
                f"| {row['id']} | {row['successes']}/{row['episodes']} | {number(row['inference_ms'])} | {number(row['speedup_inference'])} | {number(row['mean_control_steps_failure_budget'])} | {number(row['mean_control_steps_success'])} | {number(row['speedup_task_success_paper_model'])} |"
            )
        lines += [
            "",
            "Paper task times use measured request inference plus declared physical action time. Host simulator time is recorded separately. Failed episodes are charged their full declared step budget in overall statistics.",
        ]
        (output / "summary.md").write_text("\n".join(lines) + "\n")
        if args.profile_skill:
            from robotics_bench.profiling.history import build_history

            records = [
                {
                    "id": r["id"],
                    "precision": r["precision"],
                    "switches": r["config"]["switches"],
                    "samples_ms": r["inference_samples_ms"],
                    "exact": r["exact"],
                    "max_abs": r["max_abs"],
                    "recorded_at": r["recorded_at"],
                    "assessment": "Initial-observation action comparison; separate small matched task pilot, no full-suite acceptance",
                }
                for r in runs
            ]
            protocol = {
                "id": "cosmos-libero-pilot",
                "label": "Cosmos LIBERO quantization pilot",
                "workload": {
                    k: manifest[k]
                    for k in (
                        "suite",
                        "task",
                        "initial_states",
                        "sampling_seed",
                        "env_seed",
                        "steps",
                        "action_steps",
                        "paper_action_time_ms",
                    )
                },
                "environment": {
                    k: manifest[k] for k in ("gpu", "gpu_selector", "torch")
                },
                "metric": {
                    "name": "Complete policy service",
                    "unit": "ms",
                    "statistic": "median",
                },
            }
            history = build_history(
                records,
                protocol=protocol,
                cohort=output.name,
                source="runs.json",
                revision=manifest["git_head"],
                current="optimized-bf16",
            )
            for item, stats in zip(history["runs"], summary):
                item["observations"] = [
                    {
                        "name": "Pilot success rate",
                        "value": stats["success_rate"] * 100,
                        "unit": "%",
                    },
                    {
                        "name": "Mean control steps (failure budget)",
                        "value": stats["mean_control_steps_failure_budget"],
                        "unit": "steps",
                    },
                ]
                if stats["speedup_task_success_paper_model"] is not None:
                    item["observations"].append(
                        {
                            "name": "Success task speedup (paper model)",
                            "value": stats["speedup_task_success_paper_model"],
                            "unit": "x",
                        }
                    )
            (output / "ledger.json").write_text(json.dumps(history, indent=2) + "\n")
            subprocess.run(
                [
                    args.render_python,
                    str(
                        args.profile_skill.resolve()
                        / "scripts/render_optimization_history.py"
                    ),
                    str(output / "ledger.json"),
                    "--output-dir",
                    str(output / "figures"),
                    "--format",
                    "both",
                    "--update-index",
                ],
                check=True,
            )
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if simulator is not None:
            simulator.close()
        if engine is not None:
            engine.close()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
