"""Run a paired native-streaming delay study on one fixed model/simulator GPU pair."""

from pathlib import Path
import copy
import fcntl
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from robotics_bench.dynamicvla_dom import analysis, runner  # noqa: E402


def main(argv=None):
    parser = runner.parser()
    parser.description = __doc__
    parser.add_argument(
        "--task-json",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each fixed DOM scene",
    )
    parser.add_argument("--delays-ms", default="0,100,200,400,800")
    parser.add_argument("--order-seed", type=int, default=20260925)
    parser.set_defaults(
        episodes=10,
        streaming=True,
        measure_inference=True,
        episode_seed_mode=True,
        num_steps=10,
    )
    args = parser.parse_args(argv)
    delays = sorted(set(float(v) for v in args.delays_ms.split(",")))
    if not delays or 0 not in delays:
        parser.error("A zero-delay baseline is required")
    tasks = [path.expanduser().resolve() for path in args.task_json]
    if len({path.stem for path in tasks}) != len(tasks):
        parser.error("Task names must be unique")
    output = args.output_dir.expanduser().resolve()
    cells, plans = [], {}
    for task in tasks:
        for delay in delays:
            cell_id = f"{task.stem}/delay-{delay:g}"
            opts = copy.copy(args)
            opts.env_cfg = str(task)
            opts.extra_delay_ms = delay
            opts.output_dir = output / cell_id / "attempt-001"
            # Resume validation uses a nonexistent path but never writes to it.
            validate_opts = copy.copy(opts)
            validate_opts.output_dir = output / ".preflight-only"
            plan = runner.build_plan(validate_opts)
            plan["output_dir"] = str(opts.output_dir)
            cells.append(
                dict(
                    id=cell_id,
                    task=task.stem,
                    extra_delay_ms=delay,
                    num_steps=args.num_steps,
                    episodes=args.episodes,
                    seed=args.seed,
                    run_dir=str(opts.output_dir),
                )
            )
            plans[cell_id] = opts
    source = Path(__file__).resolve().parents[3]
    identity = dict(
        resources=plan["resources"],
        tasks=[str(t) for t in tasks],
        task_hashes={str(t): runner.digest(t) for t in tasks},
        delays_ms=delays,
        seed=args.seed,
        episodes=args.episodes,
        num_steps=args.num_steps,
        model_gpu=args.model_gpu,
        sim_gpu=args.sim_gpu,
        rotation=args.rotation,
        record_video=args.record_video,
        order_seed=args.order_seed,
    )
    identity["resources"].pop("env_cfg")
    identity["source_sha256"] = {
        str(p.relative_to(source)): runner.digest(p)
        for subtree in (
            "src/robotics_bench/models/dynamicvla",
            "src/robotics_bench/dynamicvla_dom",
        )
        for p in sorted((source / subtree).rglob("*.py"))
    }
    identity["source_sha256"]["src/robotics_bench/engines/dynamicvla.py"] = (
        runner.digest(source / "src/robotics_bench/engines/dynamicvla.py")
    )
    identity["checkpoint_config_sha256"] = runner.digest(
        Path(args.checkpoint or os.environ["ROBOTICS_DYNAMICVLA_CHECKPOINT"])
        / "config.json"
    )
    identity["checkpoint_sha256"] = runner.digest(
        Path(plan["resources"]["checkpoint"]) / "model.safetensors"
    )
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    try:
        code_commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = None
    study = dict(
        code_commit=code_commit,
        format=analysis.FORMAT,
        id=output.name,
        expected_tasks=[p.stem for p in tasks],
        cells=cells,
        identity=identity,
        fingerprint=fingerprint,
        design="Fixed checkpoint and streaming protocol; synthetic extra wall service delay; paired per-episode RNG and simulator seeds. No quantization or paper_sync.",
    )
    if args.dry_run:
        print(json.dumps(study, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another study supervisor holds the output lock"
            ) from error
        descriptor = output / "study.json"
        state_file = output / "state.json"
        if descriptor.exists():
            stored = runner.read(descriptor)
            if stored.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Study configuration/source changed; choose a new output directory"
                )
            study = stored
        else:
            runner.write(descriptor, study)
        state = runner.read(state_file) if state_file.exists() else dict(cells={})
        state.update(status="running", pid=os.getpid())
        runner.write(state_file, state)
        order = list(study["cells"])
        random.Random(args.order_seed).shuffle(order)
        child = None
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise KeyboardInterrupt("Study received SIGTERM")

        signal.signal(signal.SIGTERM, terminate)
        try:
            for cell in order:
                existing = Path(cell["run_dir"]) / "case-manifest.json"
                if (
                    existing.exists()
                    and runner.read(existing).get("status") == "completed"
                ):
                    analysis.audit_cell(cell, descriptor)
                    state["cells"][cell["id"]] = {
                        "status": "completed",
                        "run_dir": cell["run_dir"],
                    }
                    continue
                while True:
                    try:
                        runner.require_idle_gpus([args.model_gpu, args.sim_gpu])
                        break
                    except RuntimeError as error:
                        state["waiting"] = str(error)
                        runner.write(state_file, state)
                        print(f"Waiting for idle study GPUs: {error}", flush=True)
                        time.sleep(15)
                state.pop("waiting", None)
                parent = output / cell["id"]
                parent.mkdir(parents=True, exist_ok=True)
                attempt = 1
                while any(
                    (parent / name).exists()
                    for name in (
                        f"attempt-{attempt:03d}",
                        f"attempt-{attempt:03d}-plan.json",
                        f"attempt-{attempt:03d}.log",
                    )
                ):
                    attempt += 1
                run_dir = parent / f"attempt-{attempt:03d}"
                cell["run_dir"] = str(run_dir)
                runner.write(descriptor, study)
                opts = plans[cell["id"]]
                opts.output_dir = run_dir
                opts.env_cfg = str(next(t for t in tasks if t.stem == cell["task"]))
                plan = runner.build_plan(opts)
                # The same public runner owns both workers and their cleanup.
                plan_file = parent / f"attempt-{attempt:03d}-plan.json"
                runner.write(plan_file, plan)
                script = "from robotics_bench.dynamicvla_dom.runner import execute,read; import sys; execute(read(sys.argv[1]))"
                command = [sys.executable, "-c", script, str(plan_file)]
                env = {**os.environ, "PYTHONPATH": str(source / "src")}
                state["cells"][cell["id"]] = {
                    "status": "running",
                    "run_dir": str(run_dir),
                    "command": command,
                }
                runner.write(state_file, state)
                print(f"Start {cell['id']} ({args.episodes} episodes)", flush=True)
                with (parent / f"attempt-{attempt:03d}.log").open("x") as log:
                    child = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    code = child.wait()
                child = None
                if (
                    code
                    or runner.read(run_dir / "case-manifest.json")["status"]
                    != "completed"
                ):
                    state["cells"][cell["id"]]["status"] = "failed"
                    raise RuntimeError(
                        f"Cell failed: {cell['id']}; inspect retained attempt logs"
                    )
                state["cells"][cell["id"]]["status"] = "completed"
                runner.write(state_file, state)
                result = analysis.analyze(descriptor)
                analysis.save(result, output / "analysis")
                print(f"Completed {cell['id']}", flush=True)
            result = analysis.analyze(descriptor)
            analysis.save(result, output / "analysis")
            if result["status"] != "completed":
                raise RuntimeError("Study coverage incomplete")
            state["status"] = "completed"
            print(f"Study completed: {output}", flush=True)
        except BaseException as error:
            state.update(status="failed", error=str(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
            runner.write(state_file, state)
            signal.signal(signal.SIGTERM, previous)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
