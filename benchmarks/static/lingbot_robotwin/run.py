"""Offline LingBot + RoboTwin sync/paper_async evaluation in two local environments."""

from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_file(
    ROOT / "benchmarks/static/cosmos_libero/run.py", "_lingbot_run_helpers"
)
protocol = load_file(
    ROOT / "src/robotics_bench/protocols/lingbot_runner.py", "_lingbot_protocol"
)


def source_identity(root):
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
        if ".git" not in p.parts
    }
    result = {"path": str(root), "python_files_sha256": files}
    if (root / ".git").exists():
        command = ["git", "-c", "safe.directory=" + str(root), "-C", str(root)]
        try:
            result["git_head"] = subprocess.check_output(
                command + ["rev-parse", "HEAD"], text=True, timeout=15
            ).strip()
            result["git_dirty"] = bool(
                subprocess.check_output(
                    command + ["status", "--porcelain", "--untracked-files=no"],
                    timeout=15,
                )
            )
        except subprocess.SubprocessError as exc:
            result["git_error"] = type(exc).__name__
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, variable in {
        "lingbot-source": "ROBOTICS_LINGBOT_SOURCE",
        "robotwin-source": "ROBOTICS_ROBOTWIN_SOURCE",
        "checkpoint": "ROBOTICS_LINGBOT_CHECKPOINT",
        "model-python": "ROBOTICS_LINGBOT_PYTHON",
    }.items():
        value = os.environ.get(variable)
        parser.add_argument(
            "--" + name,
            type=Path,
            default=Path(value) if value else None,
            required=not value,
            help=f"Explicit local path; defaults to ${variable}",
        )
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument(
        "--task-config", choices=("demo_clean", "demo_randomized"), default="demo_clean"
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=10000)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-initialization-attempts", type=int, default=32)
    parser.add_argument(
        "--schedule",
        choices=("sync", "paper_async"),
        default="sync",
        help="Native sync or paper-style stale observations including KV/VAE history",
    )
    parser.add_argument(
        "--overlap-actions",
        type=int,
        default=0,
        help="n_prime in accepted control commands (0..16); sync requires zero",
    )
    parser.add_argument("--gpu", help="Physical GPU index used by model and simulator")
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_plan(args):
    paper = protocol.paper_async_contract(args.schedule, args.overlap_actions)
    if not args.task.isidentifier():
        raise ValueError("task must be a Python identifier")
    for name in (
        "episodes",
        "max_initialization_attempts",
        "video_fps",
        "startup_timeout",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.start_seed < 0 or args.model_seed < 0:
        raise ValueError("Seeds must be nonnegative")
    output = args.output_dir.expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError("Output directory already exists; choose a new directory")
    if args.gpu is not None and not args.gpu.isdigit():
        raise ValueError("This first RoboTwin case requires a physical GPU index")
    if not args.dry_run and args.gpu is None:
        raise ValueError("Actual execution requires an explicit --gpu")
    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in ("lingbot_source", "robotwin_source", "checkpoint")
    }
    # Keep the virtual-environment executable path, not its base symlink target.
    paths["model_python"] = args.model_python.expanduser().absolute()
    required = [
        paths["lingbot_source"] / "wan_va/wan_va_server.py",
        paths["lingbot_source"] / "evaluation/robotwin/msgpack_numpy.py",
        paths["robotwin_source"] / "envs" / (args.task + ".py"),
        paths["robotwin_source"] / "task_config" / (args.task_config + ".yml"),
        paths["robotwin_source"] / "task_config/_eval_step_limit.yml",
    ]
    checkpoint = paths["checkpoint"]
    required += [
        checkpoint / folder / "config.json"
        for folder in ("transformer", "vae", "text_encoder")
    ]
    required += [
        checkpoint / "tokenizer/tokenizer_config.json",
        checkpoint / "tokenizer/tokenizer.json",
        checkpoint / "vae/diffusion_pytorch_model.safetensors",
    ]
    for folder in ("transformer", "text_encoder"):
        indices = list((checkpoint / folder).glob("*.index.json"))
        if len(indices) != 1:
            raise ValueError(f"Expected one checkpoint index in {folder}")
        required += indices
        names = set(json.loads(indices[0].read_text())["weight_map"].values())
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Checkpoint shard path escapes its component")
            required.append(checkpoint / folder / name)
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(
                f"Required local file is missing or empty: {path}; downloads are disabled"
            )
    for category in ("objects", "embodiments/aloha-agilex", "background_texture"):
        if not (paths["robotwin_source"] / "assets" / category).is_dir():
            raise ValueError(f"Missing local RoboTwin assets: {category}")
    if not paths["model_python"].is_file() or not os.access(
        paths["model_python"], os.X_OK
    ):
        raise ValueError("Model Python is not an executable file")
    limits = dict(
        re.findall(
            r"^([a-zA-Z0-9_]+):\s*(\d+)\s*$",
            (paths["robotwin_source"] / "task_config/_eval_step_limit.yml").read_text(),
            re.M,
        )
    )
    if args.task not in limits:
        raise ValueError("Task has no declared native control-step budget")
    native_limit = int(limits[args.task])
    max_steps = native_limit if args.max_steps is None else args.max_steps
    if not 0 < max_steps <= native_limit:
        raise ValueError(
            "max_steps must be positive and no larger than the native task budget"
        )
    options = {
        name: getattr(args, name)
        for name in (
            "task",
            "task_config",
            "episodes",
            "start_seed",
            "model_seed",
            "max_initialization_attempts",
            "schedule",
            "overlap_actions",
            "cpu_offload",
            "record_video",
            "video_fps",
            "startup_timeout",
        )
    }
    options.update(
        max_steps=max_steps,
        frame_chunk_size=2,
        action_per_frame=16,
        keyframe_interval=4,
        step_unit="robotwin_take_action_commands",
    )
    identity = {
        "case": options,
        "observation_protocol": paper,
        "checkpoint_files": {
            str(p.relative_to(checkpoint)): p.stat().st_size
            for p in required
            if p.is_relative_to(checkpoint)
        },
        "checkpoint_sha256": {
            str(p.relative_to(checkpoint)): file_hash(p)
            for p in required
            if p.is_relative_to(checkpoint)
        },
        "entry_code_sha256": {
            p.relative_to(ROOT).as_posix(): file_hash(p)
            for p in [
                Path(__file__),
                ROOT / "src/robotics_bench/engines/lingbot.py",
                ROOT / "src/robotics_bench/engines/lingbot_server.py",
                ROOT / "src/robotics_bench/simulators/robotwin.py",
                ROOT / "src/robotics_bench/protocols/lingbot_runner.py",
                ROOT / "benchmarks/static/cosmos_libero/run.py",
                ROOT / "benchmarks/static/pi05_libero/video_recorder.py",
                ROOT / "tools/episode_statistics.py",
                ROOT / "tools/summarize_experiment.py",
            ]
        },
        "sources": {
            name: source_identity(paths[name])
            for name in ("lingbot_source", "robotwin_source")
        },
    }
    return {
        "format": "lingbot-robotwin-run-v1",
        "case_id": "lingbot_robotwin",
        "run_kind": "smoke" if args.episodes == 1 else "evaluation",
        "options": options,
        "paper_async": paper,
        "resources": {k: str(v) for k, v in paths.items()},
        "gpu": args.gpu,
        "output_dir": str(output.resolve()),
        "identity": identity,
        "case_fingerprint": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
    }


def free_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def execute(plan):
    sys.path.insert(0, str(ROOT / "src"))
    from robotics_bench.engines.lingbot import LingBotEngine
    from robotics_bench.simulators.robotwin import RoboTwinSimulator
    from robotics_bench.protocols.lingbot_runner import run_episode

    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    options, paths = plan["options"], plan["resources"]
    manifest = {
        **plan,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    common.write_json(output / "case-manifest.json", manifest)
    common.write_json(output / "paper-async.json", plan["paper_async"])
    environment = {
        "CUDA_VISIBLE_DEVICES": plan["gpu"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
    }
    old = {key: os.environ.get(key) for key in environment}
    rows, planned = [], []
    worker = engine = simulator = None
    with (
        common.console_log(output / "run.log"),
        (output / "server.log").open("x") as server_log,
    ):
        try:
            os.environ.update(environment)
            port = free_port()
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src")
                + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            }
            command = [
                paths["model_python"],
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=1",
                "--master_port=" + str(free_port()),
                "--module",
                "robotics_bench.engines.lingbot_server",
                "--source",
                paths["lingbot_source"],
                "--checkpoint",
                paths["checkpoint"],
                "--port",
                str(port),
                "--seed",
                str(options["model_seed"]),
                "--output-dir",
                str(output),
                "--cpu-offload" if options["cpu_offload"] else "--no-cpu-offload",
            ]
            print(
                "Starting the offline LingBot model worker; details: server.log",
                flush=True,
            )
            worker = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + options["startup_timeout"]
            while not (output / "server-ready.json").is_file():
                if worker.poll() is not None:
                    raise RuntimeError(
                        "LingBot model worker exited; see server.log and checkpoint-load.json"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError("LingBot model startup timed out")
                time.sleep(0.5)
            if (
                json.loads((output / "checkpoint-load.json").read_text())["status"]
                != "passed"
            ):
                raise RuntimeError("LingBot checkpoint audit did not pass")
            manifest["model_worker"] = json.loads(
                (output / "checkpoint-load.json").read_text()
            )
            manifest["simulator_python"] = sys.executable
            manifest["simulator_packages"] = {}
            for package in ("torch", "numpy", "sapien", "mplib", "websockets"):
                try:
                    manifest["simulator_packages"][package] = (
                        importlib.metadata.version(package)
                    )
                except importlib.metadata.PackageNotFoundError:
                    manifest["simulator_packages"][package] = None
            for _ in range(50):
                try:
                    engine = LingBotEngine(paths["lingbot_source"], port)
                    break
                except ConnectionRefusedError:
                    time.sleep(0.1)
            if engine is None:
                raise RuntimeError("LingBot WebSocket endpoint did not become ready")
            simulator = RoboTwinSimulator(
                paths["robotwin_source"],
                options["task"],
                task_config=options["task_config"],
                output_dir=output,
                max_initialization_attempts=options["max_initialization_attempts"],
            )
            summary = load_file(
                ROOT / "tools/summarize_experiment.py", "_lingbot_summary"
            )
            video_tools = load_file(
                ROOT / "benchmarks/static/pi05_libero/video_recorder.py",
                "_lingbot_video",
            )
            video = None
            if options["record_video"]:
                import imageio.v2 as imageio

                video = video_tools.EpisodeVideoRecorder(
                    output,
                    options["episodes"],
                    options["video_fps"],
                    lambda path, frames, fps: imageio.mimsave(path, frames, fps=fps),
                )
            next_seed = options["start_seed"]
            started = time.perf_counter()
            with (
                (output / "episodes.jsonl").open("x") as ledger,
                (output / "requests.jsonl").open("x") as requests,
            ):
                for index in range(options["episodes"]):
                    observation = simulator.prepare(next_seed, index)
                    seed = simulator.metadata["accepted_seed"]
                    if options["max_steps"] > simulator.metadata["native_step_limit"]:
                        raise RuntimeError("Native step budget differs from preflight")
                    planned.append({"task": options["task"], "init_state_id": seed})
                    manifest["planned_episodes"] = planned
                    common.write_json(output / "case-manifest.json", manifest)
                    callback = (
                        video.start(
                            [(options["task"], simulator.description, [seed])], 0
                        )
                        if video
                        else None
                    )

                    def on_frame(frame):
                        class FrameBatch:
                            def render(self):
                                return [frame]

                        callback(FrameBatch())

                    def on_event(event):
                        requests.write(
                            json.dumps(
                                {
                                    "task": options["task"],
                                    "init_state_id": seed,
                                    **event,
                                }
                            )
                            + "\n"
                        )
                        requests.flush()

                    row = run_episode(
                        engine,
                        simulator,
                        observation,
                        max_steps=options["max_steps"],
                        schedule=options["schedule"],
                        overlap_actions=options["overlap_actions"],
                        on_frame=on_frame if callback else None,
                        on_event=on_event,
                    )
                    row.update(
                        task=options["task"],
                        init_state_id=seed,
                        env_seed=seed,
                        description=simulator.description,
                        initial_observation_sha256=simulator.metadata[
                            "initial_observation_sha256"
                        ],
                    )
                    ledger.write(json.dumps(row, allow_nan=False) + "\n")
                    ledger.flush()
                    rows.append(row)
                    if video:
                        video.finish([row])
                    next_seed = seed + 1
                    common.write_json(
                        output / "coverage.json",
                        common.coverage_report(
                            planned, rows, options["max_steps"], "running"
                        ),
                    )
                    print(
                        f"Episode {index + 1}/{options['episodes']}: seed={seed} success={row['success']} commands={row['primitive_steps']}",
                        flush=True,
                    )
            results = common.evaluation_results(rows)
            results["eval_s"] = time.perf_counter() - started
            if video:
                results.update(video_paths=video.paths, video_metadata=video.records)
            common.write_json(output / "eval_results.json", results)
            common.write_json(
                output / "coverage.json",
                common.coverage_report(planned, rows, options["max_steps"], "passed"),
            )
            report = summary.write_run_summary(
                output, rows, max_steps=options["max_steps"]
            )
            print(summary.markdown_report(report))
            manifest["status"] = "completed"
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            common.write_json(
                output / "failure.json",
                {"error": manifest["error"], "traceback": traceback.format_exc()},
            )
            common.write_json(
                output / "coverage.json",
                common.coverage_report(
                    planned, rows, options["max_steps"], "failed", manifest["error"]
                ),
            )
            raise
        finally:
            for resource in (simulator, engine):
                if resource:
                    try:
                        resource.close()
                    except Exception as exc:
                        manifest.setdefault("cleanup_errors", []).append(str(exc))
            if worker is not None and worker.poll() is None:
                manifest["model_worker_shutdown"] = "case cleanup requested SIGTERM"
                try:
                    os.killpg(worker.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            common.write_json(output / "case-manifest.json", manifest)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
