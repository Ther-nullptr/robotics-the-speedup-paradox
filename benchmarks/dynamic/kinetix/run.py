"""Evaluate local RTC checkpoints on native Kinetix quality/latency cells."""

import argparse
import contextlib
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from robotics_bench.kinetix.protocol import (  # noqa: E402
    DELAY_ALIASES,
    DELAY_MODES,
    LEVELS,
    delay_plan,
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def git_head(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=15
    ).strip()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, variable in (("policy-dir", "ROBOTICS_KINETIX_POLICY_DIR"),):
        result.add_argument(
            "--" + name,
            type=Path,
            default=os.environ.get(variable),
            required=not os.environ.get(variable),
        )
    result.add_argument(
        "--levels", default="car_launch", help="Comma-separated level names, or all"
    )
    result.add_argument(
        "--flow-steps", default="5", help="Comma-separated sampler step counts"
    )
    delays = result.add_mutually_exclusive_group()
    delays.add_argument(
        "--latencies-ms", help="Comma-separated injected delays; defaults to 0"
    )
    delays.add_argument(
        "--latency-profile",
        type=Path,
        help="CSV with flow_steps,latency_ms; a declared replay scenario",
    )
    result.add_argument(
        "--mapping",
        choices=(*DELAY_MODES, *DELAY_ALIASES),
        metavar="{coarse,fine}",
        default="fine",
        help=(
            "coarse: round latency to control steps; fine: blend commands within "
            "native physics steps (default). Historical aliases remain accepted."
        ),
    )
    result.add_argument("--execute-horizon", type=int, default=4)
    result.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Paired-seed episodes per level and quality/latency cell",
    )
    result.add_argument("--start-seed", type=int, default=0)
    result.add_argument(
        "--max-steps", type=int, help="Optional smaller control-step budget"
    )
    result.add_argument("--action-noise-std", type=float, default=0.1)
    device = result.add_mutually_exclusive_group()
    device.add_argument("--gpu", type=int, help="Physical GPU index")
    device.add_argument("--cpu", action="store_true")
    result.add_argument("--record-video", action="store_true")
    result.add_argument(
        "--validate-native",
        action="store_true",
        help="Check zero-delay and constant-command equivalence before rollout",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def build_plan(args):
    if not args.dry_run and sys.version_info < (3, 11):
        raise ValueError("Kinetix execution requires Python 3.11 or newer")
    source = ROOT / "src/robotics_bench/kinetix"
    policies = args.policy_dir.expanduser().resolve()
    provenance = json.loads((source / "PROVENANCE.json").read_text())
    levels = list(LEVELS) if args.levels == "all" else args.levels.split(",")
    if (
        not levels
        or len(set(levels)) != len(levels)
        or any(level not in LEVELS for level in levels)
    ):
        raise ValueError("Select distinct supported level names or all")
    flows = [int(n) for n in args.flow_steps.split(",")]
    if not flows or min(flows) < 1 or len(set(flows)) != len(flows):
        raise ValueError("Flow steps must be distinct positive integers")
    if (
        args.episodes < 1
        or not 0 <= args.start_seed < args.start_seed + args.episodes <= 2**32
    ):
        raise ValueError("Episode seeds must form a nonempty unsigned 32-bit range")
    if not math.isfinite(args.action_noise_std) or args.action_noise_std < 0:
        raise ValueError("action_noise_std must be finite and nonnegative")
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("gpu must be nonnegative")
    if not args.dry_run and args.gpu is None and not args.cpu:
        raise ValueError("Actual execution requires --gpu INDEX or --cpu")
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output directory already exists")
    latency_source = {"kind": "specified_virtual_delay"}
    profile = {}
    if args.latency_profile:
        path = args.latency_profile.expanduser().resolve()
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                flow, latency = int(row["flow_steps"]), float(row["latency_ms"])
                if flow in profile:
                    raise ValueError(
                        "Profile must provide one latency per flow-step count"
                    )
                profile[flow] = latency
        if any(flow not in profile for flow in flows):
            raise ValueError("Profile is missing a requested flow-step count")
        latency_source = {
            "kind": "external_profile_scenario",
            "path": str(path),
            "sha256": digest(path),
            "qualification": "Input values are replayed; target-device/backend equivalence is not certified.",
        }
    else:
        delays = [float(n) for n in (args.latencies_ms or "0").split(",")]
        if len(set(delays)) != len(delays):
            raise ValueError("Latency values must be distinct")
    files = [
        source / "flow_model.py",
        source / "native/kinetix/environment/env.py",
        source / "native/jax2d/engine.py",
    ]
    tasks, cells = {}, []
    for level in levels:
        path = source / "levels" / f"{level}.json"
        checkpoint = policies / f"worlds_l_{level}.pkl"
        payload = json.loads(path.read_text())
        env, static = payload["env_params"], payload["static_env_params"]
        budget = int(env["max_timesteps"]) if args.max_steps is None else args.max_steps
        if not 0 < budget <= int(env["max_timesteps"]):
            raise ValueError("max_steps must be within every native task budget")
        tasks[level] = {
            "level_path": str(path),
            "checkpoint": str(checkpoint),
            "max_steps": budget,
            "native_env_params": env,
            "native_static_env_params": static,
        }
        files.extend([path, checkpoint])
        for flow in flows:
            for latency in [profile[flow]] if args.latency_profile else delays:
                plan = delay_plan(
                    latency,
                    physics_dt=float(env["dt"]),
                    frame_skip=int(static["frame_skip"]),
                    execute_horizon=args.execute_horizon,
                    mapping=args.mapping,
                )
                cells.append({"task": level, "flow_steps": flow, **plan})
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(
                f"Required local resource is missing: {path}; downloads are disabled"
            )
    source_files = sorted(source.rglob("*.py"))
    return {
        "format": "kinetix-native-delay-run-v1",
        "case_id": "kinetix_rtc",
        "source": str(source),
        "implementation": "repository_owned_model_environment_and_physics",
        "policy_dir": str(policies),
        "output_dir": str(output),
        "episodes_per_cell": args.episodes,
        "start_seed": args.start_seed,
        "execute_horizon": args.execute_horizon,
        "mapping": args.mapping,
        "action_noise_std": args.action_noise_std,
        "record_video": args.record_video,
        "validate_native": args.validate_native,
        "gpu": args.gpu,
        "cpu": args.cpu,
        "tasks": tasks,
        "cells": cells,
        "latency_source": latency_source,
        "protocol": {
            "world_progression": "advance_during_virtual_delay",
            "host_time_advances_simulation": False,
            "initial_previous_action": "zeros",
            "initial_policy_prefetches": 0,
            "action_noise_cadence": "control_step_shared_between_candidates",
            "physics_refinement": False,
            "blend_domain": "processed_actuator_command",
            "fractional_slot_semantics": "time_average_command_approximation",
            "seed_streams": {"environment": 0, "action_noise": 1, "policy": 2},
        },
        "resource_sha256": {str(p): digest(p) for p in files},
        "source_sha256": {str(p.relative_to(source)): digest(p) for p in source_files},
        "repository_commit": git_head(ROOT),
        "upstream_provenance": provenance,
        "entry_code_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                *sorted((ROOT / "src/robotics_bench/kinetix").glob("*.py")),
                ROOT / "tools/episode_statistics.py",
                ROOT / "src/robotics_bench/statistics.py",
            ]
        },
    }


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            if not stream.closed:
                stream.write(text)
                stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            if not stream.closed:
                stream.flush()

    def close(self):
        # Logging handlers may retain this borrowed stream until process exit.
        self.flush()


def execute(plan):
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        **plan,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "case-manifest.json", manifest)
    os.environ.update(
        JAX_PLATFORMS="cpu" if plan["cpu"] else "cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        SDL_VIDEODRIVER="dummy",
        PYTHONDONTWRITEBYTECODE="1",
        IMAGEIO_FFMPEG_NO_PREVENT_SIGINT="1",
    )
    sys.dont_write_bytecode = True
    if plan["gpu"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(plan["gpu"])
    rows = []
    with (
        (output / "run.log").open("x") as log,
        contextlib.redirect_stdout(Tee(sys.stdout, log)),
        contextlib.redirect_stderr(Tee(sys.stderr, log)),
    ):
        try:
            import jax
            import numpy as np
            from robotics_bench.kinetix.environment import KinetixEnvironment
            from robotics_bench.kinetix.policy import KinetixFlowPolicy
            from robotics_bench.kinetix.results import write_summary
            from robotics_bench.kinetix.runner import run_episode

            manifest["runtime"] = {
                "python": sys.version,
                "executable": sys.executable,
                "devices": [str(d) for d in jax.devices()],
                "packages": {},
            }
            for package in (
                "jax",
                "jaxlib",
                "flax",
                "numpy",
                "jaxgl",
                "gymnax",
                "jaxued",
                "chex",
            ):
                manifest["runtime"]["packages"][package] = importlib.metadata.version(
                    package
                )
            if not plan["cpu"] and not all(d.platform == "gpu" for d in jax.devices()):
                raise RuntimeError(
                    "Requested GPU execution but JAX did not select a GPU"
                )
            loaded = {}
            with (
                (output / "episodes.jsonl").open("x") as ledger,
                (output / "requests.jsonl").open("x") as events,
            ):
                for cell_index, cell in enumerate(plan["cells"]):
                    task = cell["task"]
                    if task not in loaded:
                        print(
                            f"Loading native Kinetix level and policy: {task}",
                            flush=True,
                        )
                        env = KinetixEnvironment(
                            task,
                            action_noise_std=plan["action_noise_std"],
                        )
                        env.reset(plan["start_seed"])
                        policy = KinetixFlowPolicy(
                            plan["tasks"][task]["checkpoint"],
                            observation_dim=env.observation_dim,
                            action_dim=env.action_dim,
                        )
                        loaded[task] = (env, policy)
                        manifest["tasks"][task]["runtime"] = {
                            "environment": env.metadata,
                            "policy": policy.metadata,
                            "model_config": asdict(policy.config),
                        }
                        if plan["validate_native"]:
                            from robotics_bench.kinetix.validation import (
                                validate_native,
                            )

                            manifest["tasks"][task]["native_validation"] = (
                                validate_native(env)
                            )
                        write_json(output / "case-manifest.json", manifest)
                    env, policy = loaded[task]
                    for seed in range(
                        plan["start_seed"],
                        plan["start_seed"] + plan["episodes_per_cell"],
                    ):
                        print(
                            f"Cell {cell_index + 1}/{len(plan['cells'])}: task={task} flow_steps={cell['flow_steps']} delay_ms={cell['requested_latency_ms']:g} seed={seed}",
                            flush=True,
                        )
                        video = None
                        frame_count = 0
                        video_shape = None
                        if plan["record_video"]:
                            import imageio.v2 as imageio

                            video_path = (
                                output
                                / "videos"
                                / f"cell-{cell_index:04d}_{task}_seed-{seed}.mp4"
                            )
                            video_path.parent.mkdir(exist_ok=True)
                            video = imageio.get_writer(
                                video_path,
                                fps=1 / cell["control_dt_seconds"],
                                macro_block_size=1,
                            )

                        def on_frame(frame):
                            nonlocal frame_count, video_shape
                            height, width = frame.shape[:2]
                            frame = np.pad(
                                frame,
                                ((0, height % 2), (0, width % 2), (0, 0)),
                                mode="edge",
                            )
                            video.append_data(frame)
                            frame_count += 1
                            video_shape = list(frame.shape)

                        def on_event(event):
                            events.write(
                                json.dumps(
                                    {
                                        "cell_index": cell_index,
                                        "task": task,
                                        "seed": seed,
                                        **event,
                                    },
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                            events.flush()

                        try:
                            row = run_episode(
                                policy,
                                env,
                                seed=seed,
                                flow_steps=cell["flow_steps"],
                                latency_ms=cell["requested_latency_ms"],
                                execute_horizon=plan["execute_horizon"],
                                mapping=plan["mapping"],
                                max_steps=plan["tasks"][task]["max_steps"],
                                on_event=on_event,
                                on_frame=on_frame if video else None,
                            )
                        finally:
                            if video:
                                video.close()
                        row.update(task=task, cell_index=cell_index)
                        if video:
                            row.update(
                                video_path=str(video_path.relative_to(output)),
                                video_frames=frame_count,
                                video_shape=video_shape,
                                video_padding="edge_to_even_dimensions",
                            )
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                        ledger.flush()
                        rows.append(row)
                        write_summary(output, rows, complete=False)
                        print(
                            f"Episode completed: success={row['success']} controls={row['primitive_steps']} reason={row['termination_reason']}",
                            flush=True,
                        )
            if len(rows) != len(plan["cells"]) * plan["episodes_per_cell"]:
                raise RuntimeError("Evaluation coverage is incomplete")
            write_summary(output, rows, complete=True)
            manifest["status"] = "completed"
            print((output / "summary.md").read_text(), flush=True)
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(
                output / "failure.json",
                {"error": manifest["error"], "traceback": traceback.format_exc()},
            )
            raise
        finally:
            manifest["completed_episodes"] = len(rows)
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(output / "case-manifest.json", manifest)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = build_plan(args)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
    else:
        execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
