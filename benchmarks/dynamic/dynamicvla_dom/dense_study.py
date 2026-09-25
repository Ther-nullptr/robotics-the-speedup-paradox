"""Expand native streaming delay studies using disjoint, randomized seed blocks."""

from pathlib import Path
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from robotics_bench.dynamicvla_dom import runner  # noqa: E402


def analyzer():
    spec = importlib.util.spec_from_file_location(
        "_dense_analysis", Path(__file__).with_name("analyze_dense_study.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser():
    p = runner.parser()
    p.description = __doc__
    p.add_argument("--task-json", type=Path, action="append", required=True)
    p.add_argument("--delays-ms", default=",".join(map(str, [*range(0, 501, 50), 800])))
    p.add_argument("--block-episodes", type=int, default=10)
    p.add_argument("--order-seed", type=int, default=20260926)
    p.add_argument(
        "--plot",
        action="store_true",
        help="Render cumulative figures after complete blocks; model Python needs matplotlib",
    )
    p.set_defaults(
        episodes=100,
        streaming=True,
        measure_inference=True,
        episode_seed_mode=True,
        num_steps=10,
    )
    return p


def plan_campaign(args):
    delays = sorted(set(float(value) for value in args.delays_ms.split(",")))
    if (
        not delays
        or 0 not in delays
        or any(not math.isfinite(value) or value < 0 for value in delays)
    ):
        raise ValueError("Delay grid must contain zero and finite nonnegative values")
    if args.episodes < 1 or args.block_episodes < 1:
        raise ValueError("Episode and block counts must be positive")
    if args.episodes % args.block_episodes:
        raise ValueError(
            "Episodes must be divisible by block-episodes for balanced blocks"
        )
    if args.seed < 0 or args.seed + args.episodes > 2**32:
        raise ValueError("Episode seeds must fit the NumPy32-bit range")
    tasks = [path.expanduser().resolve() for path in args.task_json]
    if len({path.stem for path in tasks}) != len(tasks):
        raise ValueError("Task names must be unique")
    output = args.output_dir.expanduser().resolve()
    for task in tasks:
        opts = copy.copy(args)
        opts.env_cfg = str(task)
        opts.output_dir = output / ".preflight-only" / task.stem
        checked = runner.build_plan(opts)
    resources = {
        key: value for key, value in checked["resources"].items() if key != "env_cfg"
    }
    source = Path(__file__).resolve().parents[3]
    identity = dict(
        resources=resources,
        tasks={str(path): runner.digest(path) for path in tasks},
        delays_ms=delays,
        seed=args.seed,
        episodes_per_task_condition=args.episodes,
        block_episodes=args.block_episodes,
        num_steps=args.num_steps,
        rotation=args.rotation,
        model_gpu=args.model_gpu,
        sim_gpu=args.sim_gpu,
        record_video=args.record_video,
        timeout=args.timeout,
        img_port=args.img_port,
        act_port=args.act_port,
        order_seed=args.order_seed,
    )
    identity["checkpoint_sha256"] = runner.digest(
        Path(resources["checkpoint"]) / "model.safetensors"
    )
    identity["checkpoint_config_sha256"] = runner.digest(
        Path(resources["checkpoint"]) / "config.json"
    )
    identity["source_sha256"] = {
        str(path.relative_to(source)): runner.digest(path)
        for sub in ("models/dynamicvla", "dynamicvla_dom")
        for path in sorted((source / "src/robotics_bench" / sub).rglob("*.py"))
    }
    identity["source_sha256"]["src/robotics_bench/engines/dynamicvla.py"] = (
        runner.digest(source / "src/robotics_bench/engines/dynamicvla.py")
    )
    identity["orchestrator_sha256"] = {
        name: runner.digest(Path(__file__).with_name(name))
        for name in (
            "dense_study.py",
            "analyze_dense_study.py",
            "study.py",
            "plot_study.py",
        )
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    blocks = []
    for index, offset in enumerate(range(0, args.episodes, args.block_episodes)):
        blocks.append(
            dict(
                id=f"block-{index:02d}",
                seed=args.seed + offset,
                episodes=min(args.block_episodes, args.episodes - offset),
                study=f"blocks/block-{index:02d}/study.json",
                order_seed=args.order_seed + index,
            )
        )
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(
        format="dynamicvla-dense-study-v1",
        id=output.name,
        expected_tasks=[task.stem for task in tasks],
        delays_ms=delays,
        num_steps=args.num_steps,
        episodes_per_task_condition=args.episodes,
        block_episodes=args.block_episodes,
        seed=args.seed,
        blocks=blocks,
        identity=identity,
        fingerprint=fingerprint,
        code_commit=revision,
        expected_episodes=len(tasks) * len(delays) * args.episodes,
        design="Fixed model and native streaming; disjoint paired seed blocks and randomized conditions; source and GPU pair held constant",
    ), output


def block_command(campaign, block, output):
    identity = campaign["identity"]
    resources = identity["resources"]
    command = [sys.executable, str(Path(__file__).with_name("study.py"))]
    for key in (
        "model_python",
        "sim_python",
        "checkpoint",
        "scene_dir",
        "object_dir",
        "franka_usd",
    ):
        command.extend(["--" + key.replace("_", "-"), resources[key]])
    for task in identity["tasks"]:
        command.extend(["--task-json", task])
    command.extend(
        [
            "--delays-ms",
            ",".join(map(str, campaign["delays_ms"])),
            "--episodes",
            str(block["episodes"]),
            "--seed",
            str(block["seed"]),
            "--num-steps",
            str(campaign["num_steps"]),
            "--model-gpu",
            identity["model_gpu"],
            "--sim-gpu",
            identity["sim_gpu"],
            "--rotation",
            identity["rotation"],
            "--img-port",
            str(identity["img_port"]),
            "--act-port",
            str(identity["act_port"]),
            "--timeout",
            str(identity["timeout"]),
            "--order-seed",
            str(block["order_seed"]),
            "--output-dir",
            str((output / block["study"]).parent),
        ]
    )
    if not identity["record_video"]:
        command.append("--no-record-video")
    return command


def execute(campaign, output, plot=False):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another dense-study supervisor holds the output lock"
            ) from error
        descriptor = output / "campaign.json"
        if descriptor.exists():
            stored = runner.read(descriptor)
            if stored["fingerprint"] != campaign["fingerprint"]:
                raise ValueError(
                    "Campaign identity changed; use a new output directory"
                )
            campaign = stored
        else:
            runner.write(descriptor, campaign)
        state_path = output / "state.json"
        state = runner.read(state_path) if state_path.exists() else dict(blocks={})
        state.update(
            status="running",
            pid=os.getpid(),
            started_at=state.get("started_at", datetime.now(timezone.utc).isoformat()),
        )
        state.pop("error", None)
        runner.write(state_path, state)
        helper = analyzer()
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise KeyboardInterrupt("Dense study received SIGTERM")

        signal.signal(signal.SIGTERM, terminate)
        child = None
        try:
            helper.save(helper.analyze(descriptor), output / "analysis")
            for block in campaign["blocks"]:
                block_root = (output / block["study"]).parent
                command = block_command(campaign, block, output)
                state["blocks"][block["id"]] = {
                    "status": "running",
                    "command": command,
                    "study": str(output / block["study"]),
                }
                runner.write(state_path, state)
                print(
                    f"Start {block['id']}: seeds {block['seed']}..{block['seed'] + block['episodes'] - 1}",
                    flush=True,
                )
                with (output / f"{block['id']}.log").open("a") as log:
                    child = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    code = child.wait()
                child = None
                if (
                    code
                    or runner.read(block_root / "state.json")["status"] != "completed"
                ):
                    state["blocks"][block["id"]]["status"] = "failed"
                    raise RuntimeError(
                        f"Block {block['id']} failed; inspect preserved attempts"
                    )
                state["blocks"][block["id"]]["status"] = "completed"
                runner.write(state_path, state)
                result = helper.analyze(descriptor)
                helper.save(result, output / "analysis")
                if plot:
                    subprocess.run(
                        [
                            campaign["identity"]["resources"]["model_python"],
                            str(Path(__file__).with_name("plot_study.py")),
                            "--results",
                            str(output / "analysis/results.json"),
                            "--output-dir",
                            str(output / "analysis"),
                        ],
                        check=True,
                    )
                print(
                    f"Completed {block['id']}: {result['included_episodes']}/{result['expected_episodes']} balanced episodes",
                    flush=True,
                )
            state["status"] = "completed"
        except BaseException as error:
            state.update(status="failed", error=str(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=45)
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            runner.write(state_path, state)
            signal.signal(signal.SIGTERM, previous)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    campaign, output = plan_campaign(args)
    if args.dry_run:
        print(json.dumps(campaign, indent=2))
        return 0
    return execute(campaign, output, args.plot)


if __name__ == "__main__":
    raise SystemExit(main())
