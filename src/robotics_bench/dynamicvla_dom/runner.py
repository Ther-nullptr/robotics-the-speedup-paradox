"""Offline, two-process launcher for the native single-environment DOM case."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from robotics_bench.statistics import summarize_episodes


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name, variable in (
        ("model-python", "ROBOTICS_DYNAMICVLA_PYTHON"),
        ("sim-python", "ROBOTICS_DOM_PYTHON"),
        ("checkpoint", "ROBOTICS_DYNAMICVLA_CHECKPOINT"),
        ("scene-dir", "ROBOTICS_DOM_SCENES"),
        ("object-dir", "ROBOTICS_DOM_OBJECTS"),
        ("env-cfg", "ROBOTICS_DOM_ENV_CFG"),
        ("franka-usd", "ROBOTICS_DYNAMICVLA_FRANKA_USD"),
    ):
        p.add_argument(
            "--" + name,
            default=os.environ.get(variable),
            help=f"Explicit local path, or ${variable}",
        )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--num-steps", type=int, help="Override native flow refinement steps"
    )
    p.add_argument("--rotation", choices=("euler", "rotvec", "quat"), default="euler")
    p.add_argument(
        "--streaming", action="store_true", help="Enable native DynamicVLA streaming"
    )
    p.add_argument(
        "--measure-inference",
        action="store_true",
        help="Synchronize and measure streaming model generations",
    )
    p.add_argument(
        "--extra-delay-ms",
        type=float,
        default=0,
        help="Additional worker service wall delay after CPU-ready actions",
    )
    p.add_argument(
        "--episode-seed-mode",
        action="store_true",
        help="Pair model RNG seeds with seed + episode_id",
    )
    p.add_argument("--model-gpu", default="0")
    p.add_argument("--sim-gpu", default="1")
    p.add_argument("--img-port", type=int, default=3186)
    p.add_argument("--act-port", type=int, default=3188)
    p.add_argument(
        "--record-video", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Worker startup/no-message timeout in seconds",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local paths and print commands; no GPU or output writes",
    )
    return p


def build_plan(args):
    if not math.isfinite(args.extra_delay_ms) or args.extra_delay_ms < 0:
        raise ValueError("extra-delay-ms must be finite and nonnegative")
    if (
        args.measure_inference or args.extra_delay_ms or args.episode_seed_mode
    ) and not args.streaming:
        raise ValueError("Latency-study options require streaming")
    paths = {}
    for key in (
        "model_python",
        "sim_python",
        "checkpoint",
        "scene_dir",
        "object_dir",
        "env_cfg",
        "franka_usd",
    ):
        value = getattr(args, key)
        if not value:
            raise ValueError(
                f"Missing --{key.replace('_', '-')} or matching environment variable"
            )
        # Do not dereference venv interpreter symlinks: their directory selects the environment.
        path = Path(value).expanduser().absolute()
        if not path.exists():
            raise ValueError(f"Missing local resource: {key}={path}")
        paths[key] = str(path)
    for key in ("model_python", "sim_python"):
        if not os.access(paths[key], os.X_OK):
            raise ValueError(f"Launcher is not executable: {paths[key]}")
    if args.episodes < 1 or args.seed < 0 or args.timeout <= 0:
        raise ValueError("episodes/timeout must be positive and seed nonnegative")
    if args.num_steps is not None and args.num_steps < 1:
        raise ValueError("num-steps must be positive")
    if args.img_port == args.act_port or any(
        not 1024 <= port <= 65535 for port in (args.img_port, args.act_port)
    ):
        raise ValueError("Use two different unprivileged ports")
    config = read(Path(paths["checkpoint"]) / "config.json")
    if config.get("type") != "dynamicvla":
        raise ValueError("Checkpoint must be a DynamicVLA policy")
    if not (Path(paths["checkpoint"]) / "model.safetensors").is_file():
        raise ValueError("Missing local model.safetensors")
    if not config.get("use_delta_action"):
        raise ValueError(
            "Initial DOM adapter requires the native delta-action checkpoint"
        )
    if args.streaming and config.get("chunk_size") != config.get("n_action_steps"):
        raise ValueError("Native streaming requires chunk_size == n_action_steps")
    read(paths["env_cfg"])
    output = args.output_dir.expanduser().absolute()
    if output.exists():
        raise ValueError(
            f"Output directory already exists; use a new run directory: {output}"
        )
    common = [
        "--host",
        "127.0.0.1",
        "--img-port",
        str(args.img_port),
        "--act-port",
        str(args.act_port),
        "--output-dir",
        str(output),
    ]
    launcher = [paths["sim_python"]]
    if Path(paths["sim_python"]).name == "isaaclab.sh":
        launcher.append("-p")
    server = [
        *launcher,
        "-m",
        "robotics_bench.dynamicvla_dom.server",
        *common,
        "--scene-dir",
        paths["scene_dir"],
        "--object-dir",
        paths["object_dir"],
        "--env-cfg",
        paths["env_cfg"],
        "--franka-usd",
        paths["franka_usd"],
        "--seed",
        str(args.seed),
        "--episodes",
        str(args.episodes),
        "--device",
        "cuda:0",
        "--headless",
        "--enable_cameras",
    ]
    if args.measure_inference:
        server.append("--allow-policy-starvation")
    if args.record_video:
        server.append("--record-video")
    client = [
        paths["model_python"],
        "-m",
        "robotics_bench.dynamicvla_dom.client",
        *common,
        "--checkpoint",
        paths["checkpoint"],
        "--device",
        "cuda:0",
        "--seed",
        str(args.seed),
        "--episodes",
        str(args.episodes),
        "--timeout",
        str(args.timeout),
        "--rotation",
        args.rotation,
    ]
    if args.streaming:
        client.append("--streaming")
    if args.measure_inference:
        client.append("--measure-inference")
    if args.episode_seed_mode:
        client.append("--episode-seed-mode")
    client.extend(["--extra-delay-ms", str(args.extra_delay_ms)])
    if args.num_steps is not None:
        client.extend(["--num-steps", str(args.num_steps)])
    source = Path(__file__).resolve().parents[2]
    env = {
        "PYTHONPATH": str(source),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    options = dict(
        measure_inference=args.measure_inference,
        extra_delay_ms=args.extra_delay_ms,
        episode_seed_mode=args.episode_seed_mode,
        mode="native-streaming" if args.streaming else "native-non-streaming",
        episodes=args.episodes,
        seed=args.seed,
        rotation=args.rotation,
        num_steps=args.num_steps
        if args.num_steps is not None
        else config.get("num_steps"),
        n_action_steps=config["n_action_steps"],
        action_horizon=config["chunk_size"],
        n_obs_steps=config["n_obs_steps"],
        record_video=args.record_video,
        model_gpu=args.model_gpu,
        sim_gpu=args.sim_gpu,
    )
    return dict(
        case_id="dynamicvla_dom",
        status="planned",
        options=options,
        resources=paths,
        output_dir=str(output),
        commands=dict(server=server, client=client),
        environments=dict(
            client={**env, "CUDA_VISIBLE_DEVICES": args.model_gpu},
            server={**env, "CUDA_VISIBLE_DEVICES": args.sim_gpu},
        ),
        identity={
            "case": options,
            "checkpoint_config_sha256": digest(
                Path(paths["checkpoint"]) / "config.json"
            ),
            "task_config_sha256": digest(paths["env_cfg"]),
        },
    )


def require_idle_gpus(selectors):
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    records = [
        tuple(value.strip() for value in row.split(",")) for row in rows.splitlines()
    ]
    for selector in set(selectors):
        matched = [row for row in records if selector in row[:2]]
        if len(matched) != 1:
            raise ValueError(f"Unknown single GPU selector: {selector}")
        _, uuid, usage, memory = matched[0]
        if uuid in processes or int(usage) != 0 or int(memory) > 200:
            raise RuntimeError(
                f"GPU {selector} is not clean and idle; wait or choose another GPU"
            )
    return records


def lines(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def summarize(output, expected_episodes):
    output = Path(output)
    episodes = lines(output / "episodes.jsonl")
    if len(episodes) != expected_episodes or {
        row["episode_id"] for row in episodes
    } != set(range(expected_episodes)):
        raise ValueError("Incomplete episode coverage")
    generated = defaultdict(int)
    seen = set()
    for event in lines(output / "requests.jsonl"):
        if event.get("kind") != "chunk_generated":
            continue
        key = event["episode_id"], event["chunk_id"]
        if key in seen or event["episode_id"] not in range(expected_episodes):
            raise ValueError("Duplicate or unknown chunk event identity")
        seen.add(key)
        generated[event["episode_id"]] += 1
    for row in episodes:
        if generated[row["episode_id"]] == 0:
            raise ValueError("Episode has no completed model generation evidence")
        if "applied_chunk_ids" in row:
            ids = row["applied_chunk_ids"]
            if len(set(ids)) != len(ids) or row["applied_chunks"] != len(ids):
                raise ValueError("Applied chunk ledger is inconsistent")
            if any((row["episode_id"], chunk_id) not in seen for chunk_id in ids):
                raise ValueError("Applied chunk has no matching model generation event")
    budgets = {row["task"]: row["max_primitive_steps"] for row in episodes}
    if any(row["max_primitive_steps"] != budgets[row["task"]] for row in episodes):
        raise ValueError("Per-task episode budgets differ")
    stats = summarize_episodes(episodes, task_max_steps=budgets)["overall"]
    successful = [row for row in episodes if row["success"]]
    result = {
        **stats,
        "mean_chunks_success": sum(generated[row["episode_id"]] for row in successful)
        / len(successful)
        if successful
        else None,
        "generated_chunks": sum(generated.values()),
        "mean_applied_chunks_success": sum(row["applied_chunks"] for row in successful)
        / len(successful)
        if successful and all("applied_chunks" in row for row in successful)
        else None,
        "chunk_definition": "Actual completed model generation; includes generated chunks not executed by DOM. Cache pops and held control steps are separate.",
        "inference_timing": "requests.jsonl model events; action selection and transport are not pure inference timing",
    }
    write(output / "summary.json", result)
    write(
        output / "coverage.json",
        dict(
            status="passed",
            expected_episodes=expected_episodes,
            completed_episodes=len(episodes),
            successes=stats["successes"],
        ),
    )

    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"

    (output / "summary.md").write_text(
        f"# DynamicVLA + DOM native evaluation\n\nEpisodes: {len(episodes)}\n\nSuccess rate: {stats['success_rate']:.6f}\n\nMean generated chunks (successful episodes): {fmt(result['mean_chunks_success'])}\n\nMean applied chunks (successful episodes): {fmt(result['mean_applied_chunks_success'])}\n\nMean control steps (failure budget): {fmt(stats['failure_penalized']['mean_control_steps'])}\n\nMean control steps (successful episodes): {fmt(stats['successful_episodes']['mean_control_steps'])}\n\n{result['chunk_definition']}\n"
    )
    return result


def stop(process):
    # The owned process group may outlive its leader (streaming worker/manager).
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(plan):
    gpu_snapshot = require_idle_gpus(
        [plan["options"]["model_gpu"], plan["options"]["sim_gpu"]]
    )
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        **plan,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gpu_preflight": gpu_snapshot,
    }
    source = Path(__file__).resolve().parents[1]
    files = (
        list((source / "dynamicvla_dom").rglob("*.py"))
        + list((source / "models/dynamicvla").rglob("*.py"))
        + [source / "engines/dynamicvla.py"]
    )
    manifest["source_sha256"] = {
        str(path.relative_to(source)): digest(path) for path in files
    }
    manifest["checkpoint_sha256"] = digest(
        Path(plan["resources"]["checkpoint"]) / "model.safetensors"
    )
    write(output / "case-manifest.json", manifest)
    processes = {}
    streams = []
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt("Launcher received SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        for role in ("server", "client"):
            stream = (output / f"{role}.log").open("x")
            streams.append(stream)
            env = {**os.environ, **plan["environments"][role]}
            if role == "server" and Path(plan["resources"]["sim_python"]).name in (
                "isaaclab.sh",
                "python.sh",
            ):
                for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME"):
                    env.pop(key, None)
            processes[role] = subprocess.Popen(
                plan["commands"][role],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        while any(process.poll() is None for process in processes.values()):
            for role, process in processes.items():
                status_path = output / f"{role}-status.json"
                if status_path.exists() and read(status_path).get("status") == "failed":
                    raise RuntimeError(
                        f"{role} reported failure: {read(status_path).get('error')}"
                    )
                if process.poll() not in (None, 0):
                    raise RuntimeError(
                        f"{role} exited with {process.returncode}; see {output / (role + '.log')}"
                    )
            time.sleep(0.5)
        for role, process in processes.items():
            if process.returncode != 0:
                raise RuntimeError(
                    f"{role} exited with {process.returncode}; inspect worker log"
                )
        if read(output / "client-status.json")["status"] != "completed":
            raise RuntimeError("Client did not complete")
        if read(output / "server-status.json")["status"] != "completed":
            raise RuntimeError("Server did not complete")
        summarize(output, plan["options"]["episodes"])
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        for process in processes.values():
            stop(process)
        for stream in streams:
            stream.close()
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "case-manifest.json", manifest)
        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            execute(plan)
            print(f"Completed: {plan['output_dir']}")
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
