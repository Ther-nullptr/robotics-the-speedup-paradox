"""Run a paired flow-step quality sweep without changing native simulation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from robotics_bench.kinetix.protocol import LEVELS  # noqa: E402


def make_jobs(
    *, python, policy_dir, output_dir, levels, flow_steps, episodes, start_seed
):
    if (
        not levels
        or len(set(levels)) != len(levels)
        or any(t not in LEVELS for t in levels)
    ):
        raise ValueError("Select distinct supported levels")
    if not flow_steps or len(set(flow_steps)) != len(flow_steps) or min(flow_steps) < 1:
        raise ValueError("Flow steps must be distinct positive integers")
    if episodes < 1 or not 0 <= start_seed < start_seed + episodes <= 2**32:
        raise ValueError("Invalid paired seed range")
    jobs = []
    for level in levels:
        args = [
            "--policy-dir",
            str(policy_dir),
            "--levels",
            level,
            "--flow-steps",
            ",".join(map(str, flow_steps)),
            "--episodes",
            str(episodes),
            "--start-seed",
            str(start_seed),
            "--latencies-ms",
            "0",
            "--mapping",
            "native-blend",
            "--execute-horizon",
            "4",
            "--action-noise-std",
            "0.1",
            "--validate-native",
            "--output-dir",
            str(Path(output_dir) / level),
        ]
        jobs.append(
            {
                "task": level,
                "output_dir": str(Path(output_dir) / level),
                "args": args,
                "command_prefix": [
                    str(python),
                    "-B",
                    "-u",
                    str(Path(__file__).with_name("run.py")),
                ],
            }
        )
    return jobs


def preflight(jobs):
    spec = importlib.util.spec_from_file_location(
        "kinetix_native_case", Path(__file__).with_name("run.py")
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    for job in jobs:
        entry.build_plan(entry.parser().parse_args([*job["args"], "--dry-run"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policy-dir",
        type=Path,
        default=os.environ.get("ROBOTICS_KINETIX_POLICY_DIR"),
        required=not os.environ.get("ROBOTICS_KINETIX_POLICY_DIR"),
    )
    p.add_argument(
        "--python",
        type=Path,
        default=os.environ.get("ROBOTICS_KINETIX_PYTHON", sys.executable),
    )
    p.add_argument("--levels", default="all")
    p.add_argument("--flow-steps", default="1,2,3,4,5")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--gpus", default="1,2,3")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    levels = list(LEVELS) if args.levels == "all" else args.levels.split(",")
    flows = [int(n) for n in args.flow_steps.split(",")]
    gpus = [int(n) for n in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        p.error("Select distinct nonnegative GPU indices")
    output = args.output_dir.expanduser().absolute()
    if output.exists():
        p.error("Output directory already exists; use a new directory")
    interpreter = shutil.which(str(args.python.expanduser()))
    if interpreter is None:
        p.error("Python interpreter was not found or is not executable")
    jobs = make_jobs(
        python=Path(interpreter).absolute(),
        policy_dir=args.policy_dir.expanduser().resolve(),
        output_dir=output,
        levels=levels,
        flow_steps=flows,
        episodes=args.episodes,
        start_seed=args.start_seed,
    )
    preflight(jobs)
    plan = {
        "format": "kinetix-flow-quality-sweep-v1",
        "status": "planned",
        "levels": levels,
        "flow_steps": flows,
        "episodes_per_cell": args.episodes,
        "start_seed": args.start_seed,
        "expected_episodes": len(levels) * len(flows) * args.episodes,
        "gpus": gpus,
        "protocol": {
            "physics_dt_seconds": 1 / 60,
            "frame_skip": 2,
            "execute_horizon": 4,
            "action_noise_std": 0.1,
            "injected_latency_ms": 0,
            "control_step_budget": 256,
        },
        "jobs": jobs,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    lock, pending = threading.Lock(), queue.Queue()
    for job in jobs:
        pending.put(job)
    start = time.perf_counter()
    plan.update(status="running", started_at=datetime.now(timezone.utc).isoformat())

    def save():
        temp = output / "sweep-manifest.json.tmp"
        temp.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        temp.replace(output / "sweep-manifest.json")

    save()

    def worker(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            command = [*job["command_prefix"], *job["args"], "--gpu", str(gpu)]
            started = time.perf_counter()
            with lock:
                job.update(status="running", gpu=gpu, command=command)
                save()
            print(f"Starting {job['task']} on GPU {gpu}", flush=True)
            try:
                with (output / "logs" / f"{job['task']}.log").open("x") as log:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                code = result.returncode
                if code == 0:
                    result_manifest = json.loads(
                        (Path(job["output_dir"]) / "case-manifest.json").read_text()
                    )
                    if (
                        result_manifest["status"] != "completed"
                        or result_manifest["completed_episodes"]
                        != len(flows) * args.episodes
                    ):
                        raise ValueError(
                            "Child exited without complete evaluation coverage"
                        )
                error = None
            except Exception as exc:
                code = -1
                error = f"{type(exc).__name__}: {exc}"
            with lock:
                job.update(
                    status="completed" if code == 0 else "failed",
                    returncode=code,
                    host_seconds=time.perf_counter() - started,
                )
                if error:
                    job["error"] = error
                save()
            print(
                f"Finished {job['task']}: status={job['status']} wall_seconds={job['host_seconds']:.3f}",
                flush=True,
            )
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    plan.update(
        status="completed"
        if all(j["status"] == "completed" for j in jobs)
        else "failed",
        host_seconds=time.perf_counter() - start,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    save()
    if plan["status"] != "completed":
        raise SystemExit(
            "One or more tasks failed; inspect logs and do not treat this as a complete sweep"
        )
    print(f"Completed {plan['expected_episodes']} episodes. Results: {output}")


if __name__ == "__main__":
    main()
