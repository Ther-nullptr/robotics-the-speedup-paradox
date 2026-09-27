"""Plan and run paired native-physics latency studies with explicit data reuse."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from robotics_bench.kinetix.protocol import (  # noqa: E402
    LEVELS,
    delay_plan,
    normalize_mapping,
)

DEFAULT_ARCHIVE = ROOT / "docs/data/kinetix-hardware-latencies-20260811.csv"
DEFAULT_TASKS = ("mjc_walker", "catapult", "trampoline", "catcher_v3")
HARDWARE = ("local_ada", "rtx3090", "agx15", "agx30")
SOURCE = ROOT / "src/robotics_bench/kinetix"
# Audited naming-only migration. Other source differences prohibit data reuse.
NAMING_MIGRATION = {
    "protocol.py": (
        "70f176c6c2817ff570fc0c72465f4887214fee2d9b0ab9922682be1040184ed8",
        "dfdeb00afdec9ca955bfe4617839a6aacb1a2b1b69cb4a74dcdda16c91df8e7d",
    ),
    "runner.py": (
        "9b48f5258eded711aba77e1d2633c4c68bd2b1204034c1d9a2af3202ef2d4c59",
        "a887c9f4f5c22a9bc209a9767c432d2632edbbdeec4e312080d54147ea45e471",
    ),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_profiles(path):
    profiles = {name: {} for name in HARDWARE}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            hardware = row["profile_id"]
            if hardware not in profiles:
                raise ValueError(f"Unsupported hardware profile: {hardware}")
            n, latency = int(row["flow_steps"]), float(row["latency_ms"])
            if n in profiles[hardware]:
                raise ValueError("Duplicate hardware/flow-step profile row")
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("Profile latency must be finite and nonnegative")
            profiles[hardware][n] = latency
    if any(set(values) != set(range(1, 6)) for values in profiles.values()):
        raise ValueError("Each hardware profile must contain exactly N=1..5")
    return profiles


def weight_key(task, flow, plan):
    """Identity at the actual float32 command-weight boundary of run_episode."""
    weights = b"".join(
        struct.pack("!f", value) for pair in plan["old_weights"] for value in pair
    )
    return f"{task}/N{flow}/{weights.hex()}"


def make_matrix(tasks, profiles):
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(LEVELS):
        raise ValueError("Select distinct supported tasks")
    conditions, executions, jobs, sources = [], {}, {}, {}

    def add(task, role, mode, flow, latency, hardware="none"):
        plan = delay_plan(
            latency, physics_dt=1 / 60, frame_skip=2, execute_horizon=4, mapping=mode
        )
        key = weight_key(task, flow, plan)
        if key not in sources:
            if role == "quality":
                source = {"kind": "baseline", "id": f"{task}/N{flow}"}
            else:
                execution_id = hashlib.sha256(key.encode()).hexdigest()[:16]
                if role == "calibration":
                    group = f"{task}/delay"
                elif mode == "fine":
                    group = f"{task}/hardware-{hardware}"
                else:
                    group = f"{task}/coarse"
                execution = {
                    "id": execution_id,
                    "task": task,
                    "flow_steps": flow,
                    "mode": mode,
                    "latency_ms": (
                        plan["effective_latency_ms"] if mode == "coarse" else latency
                    ),
                    "effective_latency_ms": plan["effective_latency_ms"],
                    "weight_key": key,
                    "job": group,
                    "phase": "calibration" if role == "calibration" else "validation",
                }
                executions[execution_id] = execution
                job = jobs.setdefault(
                    group,
                    {
                        "id": group,
                        "task": task,
                        "mode": mode,
                        "phase": execution["phase"],
                        "hardware": hardware,
                        "execution_ids": [],
                    },
                )
                job["execution_ids"].append(execution_id)
                source = {"kind": "execution", "id": execution_id}
            sources[key] = (source, role)
        source, original_role = sources[key]
        conditions.append(
            {
                "id": f"c{len(conditions):04d}",
                "task": task,
                "role": role,
                "mode": mode,
                "flow_steps": flow,
                "hardware": hardware,
                "requested_latency_ms": latency,
                "effective_latency_ms": plan["effective_latency_ms"],
                "old_weights": plan["old_weights"],
                "source": source,
                "calibration_overlap": role == "validation"
                and original_role in {"quality", "calibration"},
            }
        )

    for task in tasks:
        for n in range(1, 6):
            add(task, "quality", "fine", n, 0.0, "zero_delay")
        for k in range(9):
            add(task, "calibration", "fine", 5, k * (1000 / 120))
        for k in range(3):
            add(task, "calibration", "coarse", 5, k * (1000 / 30))
        for mode in ("fine", "coarse"):
            for hardware, values in profiles.items():
                for n, latency in values.items():
                    add(task, "validation", mode, n, latency, hardware)
    return {
        "conditions": conditions,
        "executions": list(executions.values()),
        "jobs": list(jobs.values()),
    }


def check_coverage(rows, flows, *, start_seed, episodes):
    keys = [(int(row["flow_steps"]), int(row["env_seed"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate episode key")
    expected = {
        (flow, seed)
        for flow in flows
        for seed in range(start_seed, start_seed + episodes)
    }
    if set(keys) != expected:
        raise ValueError("Episode coverage differs from the declared seed/flow grid")


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_baseline(directory, tasks, policy_dir, *, episodes, start_seed):
    """Require a complete matched zero-delay grid before importing any scores."""
    evidence = {}
    current_sources = {
        str(path.relative_to(SOURCE)): digest(path) for path in SOURCE.rglob("*.py")
    }
    for task in tasks:
        manifest_path = directory / task / "case-manifest.json"
        ledger_path = directory / task / "episodes.jsonl"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["status"] != "completed"
            or manifest["episodes_per_cell"] != episodes
            or manifest["start_seed"] != start_seed
            or manifest["execute_horizon"] != 4
            or manifest["action_noise_std"] != 0.1
            or normalize_mapping(manifest["mapping"]) != "fine"
        ):
            raise ValueError(f"Incompatible baseline configuration: {task}")
        old_sources = manifest["source_sha256"]
        if set(old_sources) != set(current_sources):
            raise ValueError("Baseline runtime source inventory changed")
        for name, old_hash in old_sources.items():
            current_hash = current_sources[name]
            if old_hash != current_hash and NAMING_MIGRATION.get(name) != (
                old_hash,
                current_hash,
            ):
                raise ValueError(f"Unverified baseline source change: {name}")
        native = json.loads((SOURCE / "levels" / f"{task}.json").read_text())
        task_meta = manifest["tasks"][task]
        if (
            task_meta["native_env_params"] != native["env_params"]
            or task_meta["native_static_env_params"] != native["static_env_params"]
            or task_meta["max_steps"] != 256
        ):
            raise ValueError(f"Baseline native parameters differ: {task}")
        if (
            native["env_params"]["dt"] != 1 / 60
            or native["static_env_params"]["frame_skip"] != 2
        ):
            raise ValueError("Study requires native 60 Hz physics / 30 Hz control")
        resources = manifest["resource_sha256"]
        checkpoint = policy_dir / f"worlds_l_{task}.pkl"
        if resources[task_meta["checkpoint"]] != digest(checkpoint):
            raise ValueError(f"Baseline checkpoint differs: {task}")
        if resources[task_meta["level_path"]] != digest(
            SOURCE / "levels" / f"{task}.json"
        ):
            raise ValueError(f"Baseline level differs: {task}")
        protocol = manifest["protocol"]
        if (
            protocol["host_time_advances_simulation"]
            or protocol["initial_policy_prefetches"] != 0
            or protocol["initial_previous_action"] != "zeros"
            or protocol["physics_refinement"]
            or protocol["blend_domain"] != "processed_actuator_command"
        ):
            raise ValueError("Baseline delay protocol differs")
        rows = read_jsonl(ledger_path)
        check_coverage(rows, range(1, 6), start_seed=start_seed, episodes=episodes)
        for row in rows:
            if (
                row["task"] != task
                or row["requested_latency_ms"] != 0
                or row["effective_latency_ms"] != 0
                or normalize_mapping(row["delay_mapping"]) != "fine"
                or type(row["success"]) is not bool
                or row["max_primitive_steps"] != 256
                or not 1 <= row["primitive_steps"] <= 256
                or row["physics_steps"] != row["primitive_steps"] * 2
            ):
                raise ValueError(f"Invalid baseline episode: {task}")
        for source in manifest.get("composition", {}).get("sources", []):
            source_dir = Path(source["directory"])
            for name, field in (
                ("case-manifest.json", "manifest_sha256"),
                ("episodes.jsonl", "episodes_sha256"),
            ):
                if digest(source_dir / name) != source[field]:
                    raise ValueError(
                        f"Baseline composition source changed: {source_dir}"
                    )
        evidence[task] = {
            "manifest": str(manifest_path),
            "manifest_sha256": digest(manifest_path),
            "episodes": str(ledger_path),
            "episodes_sha256": digest(ledger_path),
            "checkpoint_sha256": digest(checkpoint),
            "runtime_packages": manifest["runtime"]["packages"],
            "model_config": task_meta["runtime"]["model_config"],
        }
    return evidence


def create_study(args):
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError("Study directory already exists; use run to resume it")
    tasks = args.tasks.split(",")
    profiles = read_profiles(args.hardware_profiles)
    matrix = make_matrix(tasks, profiles)
    if (
        args.episodes < 2
        or not 0 <= args.start_seed < args.start_seed + args.episodes <= 2**32
    ):
        raise ValueError("Use at least two paired seeds within unsigned32 range")
    baseline = verify_baseline(
        args.quality_run.expanduser().resolve(),
        tasks,
        args.policy_dir.expanduser().resolve(),
        episodes=args.episodes,
        start_seed=args.start_seed,
    )
    manifest = {
        "format": "kinetix-latency-study-v1",
        "created_at": utc_now(),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [*SOURCE.rglob("*.py"), Path(__file__).with_name("run.py")]
        },
        "tasks": tasks,
        "episodes_per_cell": args.episodes,
        "start_seed": args.start_seed,
        "policy_dir": str(args.policy_dir.expanduser().resolve()),
        "quality_run": str(args.quality_run.expanduser().resolve()),
        "baseline": baseline,
        "hardware_profiles": profiles,
        "hardware_profile_sha256": digest(args.hardware_profiles),
        "hardware_interpretation": "Historical PyTorch FP16 architecture latency replay with a trained JAX FP32 policy; not hardware-in-the-loop.",
        "validation_scope": "Held-out conditions with paired seeds, not held-out random seeds or scene initializations.",
        "selection_basis": "Existing zero-delay quality results, chosen before new latency outcomes.",
        **matrix,
    }
    manifest["verification_jobs"] = [
        {
            "id": f"{task}/verify-{mode}",
            "task": task,
            "mode": mode,
            "phase": "verification",
            "execution_ids": [],
            "hardware": "none",
        }
        for task in tasks
        for mode in ("fine", "coarse")
    ]
    output.mkdir(parents=True)
    (output / "profiles").mkdir()
    for hardware, values in profiles.items():
        with (output / "profiles" / f"{hardware}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["flow_steps", "latency_ms"])
            writer.writerows(values.items())
    write_json(output / "study.json", manifest)
    print(
        json.dumps(
            {
                "study": str(output),
                "tasks": tasks,
                "logical_cells": len(matrix["conditions"]),
                "unique_new_cells": len(matrix["executions"]),
                "new_episodes": len(matrix["executions"]) * args.episodes,
                "reused_zero_delay_episodes": len(tasks) * 5 * args.episodes,
                "jobs": len(matrix["jobs"]),
            },
            indent=2,
        )
    )


def verify_inputs(study):
    for relative, expected in study["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Runtime changed after study planning: {relative}")
    for task, evidence in study["baseline"].items():
        for name in ("manifest", "episodes"):
            if digest(evidence[name]) != evidence[f"{name}_sha256"]:
                raise ValueError(f"Baseline {name} changed: {task}")
        if (
            digest(Path(study["policy_dir"]) / f"worlds_l_{task}.pkl")
            != evidence["checkpoint_sha256"]
        ):
            raise ValueError(f"Checkpoint changed: {task}")


def job_command(study, job, directory, *, python, gpu, output):
    count = 2 if job["phase"] == "verification" else study["episodes_per_cell"]
    command = [
        str(python),
        str(Path(__file__).with_name("run.py")),
        "--policy-dir",
        study["policy_dir"],
        "--levels",
        job["task"],
        "--mapping",
        job["mode"],
        "--episodes",
        str(count),
        "--start-seed",
        str(study["start_seed"]),
        "--gpu",
        str(gpu),
        "--output-dir",
        str(output),
    ]
    if job["phase"] == "verification":
        return command + [
            "--flow-steps",
            "1,2,3,4,5",
            "--latencies-ms",
            "0",
            "--validate-native",
        ]
    executions = {row["id"]: row for row in study["executions"]}
    cells = [executions[key] for key in job["execution_ids"]]
    flows = sorted({row["flow_steps"] for row in cells})
    command += ["--flow-steps", ",".join(map(str, flows))]
    if job["mode"] == "fine" and job["phase"] == "validation":
        profile = directory / "profiles" / f"{job['hardware']}.csv"
        expected = study["hardware_profiles"][job["hardware"]]
        with profile.open(newline="") as stream:
            actual = {
                r["flow_steps"]: float(r["latency_ms"]) for r in csv.DictReader(stream)
            }
        if actual != {str(k): v for k, v in expected.items()}:
            raise ValueError("Exported latency profile changed")
        return command + ["--latency-profile", str(profile)]
    latencies = sorted({row["latency_ms"] for row in cells})
    if len(flows) * len(latencies) != len(cells):
        raise ValueError("Job grouping would create unintended cross-product cells")
    return command + ["--latencies-ms", ",".join(map(repr, latencies))]


def verify_job(study, job, output):
    manifest = json.loads((output / "case-manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError(f"Incomplete job: {job['id']}")
    expected_sources = {
        key.removeprefix("src/robotics_bench/kinetix/"): value
        for key, value in study["source_sha256"].items()
        if key.startswith("src/robotics_bench/kinetix/")
    }
    if manifest["source_sha256"] != expected_sources:
        raise ValueError("Executed runtime source differs from the frozen study")
    expected_episodes = (
        2 if job["phase"] == "verification" else study["episodes_per_cell"]
    )
    if (
        manifest["episodes_per_cell"] != expected_episodes
        or manifest["start_seed"] != study["start_seed"]
        or manifest["action_noise_std"] != 0.1
        or manifest["execute_horizon"] != 4
        or normalize_mapping(manifest["mapping"]) != job["mode"]
        or manifest["protocol"]["host_time_advances_simulation"]
    ):
        raise ValueError("Executed configuration differs from the study")
    if (
        manifest["runtime"]["packages"]
        != study["baseline"][job["task"]]["runtime_packages"]
    ):
        raise ValueError("Runtime package versions differ from the zero-delay baseline")
    task = job["task"]
    if (
        manifest["tasks"][task]["runtime"]["model_config"]
        != study["baseline"][task]["model_config"]
    ):
        raise ValueError("Runtime model configuration differs from baseline")
    rows = read_jsonl(output / "episodes.jsonl")
    if job["phase"] == "verification":
        check_coverage(rows, range(1, 6), start_seed=study["start_seed"], episodes=2)
        baseline = {
            (r["flow_steps"], r["env_seed"]): r
            for r in read_jsonl(study["baseline"][task]["episodes"])
        }
        fields = (
            "success",
            "primitive_steps",
            "physics_steps",
            "inference_calls",
            "episode_return",
            "termination_reason",
            "initial_observation_sha256",
        )
        for row in rows:
            prior = baseline[(row["flow_steps"], row["env_seed"])]
            if any(row[field] != prior[field] for field in fields):
                raise ValueError(
                    f"Zero-delay parity failed: {task}, seed={row['env_seed']}, N={row['flow_steps']}"
                )
    else:
        cells = {row["id"]: row for row in study["executions"]}
        grouped = {}
        for row in rows:
            plan = delay_plan(
                row["requested_latency_ms"],
                physics_dt=1 / 60,
                frame_skip=2,
                execute_horizon=4,
                mapping=row["delay_mapping"],
            )
            key = weight_key(task, row["flow_steps"], plan)
            grouped.setdefault(key, []).append(row)
        expected = {cells[key]["weight_key"] for key in job["execution_ids"]}
        if set(grouped) != expected:
            raise ValueError("Executed job cells do not match study plan")
        for group in grouped.values():
            check_coverage(
                group,
                [group[0]["flow_steps"]],
                start_seed=study["start_seed"],
                episodes=study["episodes_per_cell"],
            )
    return {
        "episodes": len(rows),
        "manifest_sha256": digest(output / "case-manifest.json"),
        "episodes_sha256": digest(output / "episodes.jsonl"),
    }


def gpu_is_empty(gpu):
    status = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return not processes and float(status[0]) <= 5 and float(status[1]) <= 500


def run_phase(args):
    import fcntl

    directory = args.study_dir.expanduser().resolve()
    study = json.loads((directory / "study.json").read_text())
    verify_inputs(study)
    gpus = [int(value) for value in args.gpus.split(",")]
    if (
        not gpus
        or min(gpus) < 0
        or len(gpus) != len(set(gpus))
        or args.cpus_per_worker < 1
    ):
        raise ValueError("Choose distinct nonnegative GPU indices")
    lock_stream = (directory / "controller.lock").open("a+")
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = directory / "status.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {"jobs": {}, "phases": {}}
    )
    if (
        args.phase != "verification"
        and status["phases"].get("verification") != "completed"
    ):
        raise ValueError("Complete zero-delay verification before experiments")
    if args.phase == "validation":
        if status["phases"].get("calibration") != "completed":
            raise ValueError("Complete calibration before validation")
        frozen = directory / "calibration" / "frozen-model.json"
        if not frozen.is_file():
            raise ValueError("Freeze calibration predictions before mapped validation")
        freeze = json.loads(frozen.read_text())
        if freeze["study_sha256"] != digest(directory / "study.json"):
            raise ValueError("Frozen model belongs to a different study")
        previous = status.get("frozen_model_sha256")
        if previous and previous != digest(frozen):
            raise ValueError("Frozen model changed after validation started")
        status["frozen_model_sha256"] = digest(frozen)
    jobs = (
        study["verification_jobs"]
        if args.phase == "verification"
        else [job for job in study["jobs"] if job["phase"] == args.phase]
    )
    pending = []
    for job in jobs:
        record = status["jobs"].get(job["id"], {})
        artifact = Path(record["output"]) / "case-manifest.json" if record else None
        completed_artifact = (
            artifact
            and artifact.exists()
            and json.loads(artifact.read_text()).get("status") == "completed"
        )
        if record.get("status") == "completed" or completed_artifact:
            verified = verify_job(study, job, Path(record["output"]))
            if record.get("verification") and verified != record["verification"]:
                raise ValueError("Completed job artifact changed")
            record.update(status="completed", verification=verified)
        else:
            pending.append(job)
    status.update(controller_pid=os.getpid(), phase=args.phase, updated_at=utc_now())
    status["phases"][args.phase] = "running"
    write_json(status_path, status)
    mutex, stopped = threading.Lock(), threading.Event()
    available_cpus = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )

    def worker(lane, gpu):
        while not stopped.is_set():
            with mutex:
                if not pending:
                    return
                job = pending.pop(0)
            while not gpu_is_empty(gpu):
                if stopped.wait(10):
                    return
            with mutex:
                record = status["jobs"].get(job["id"], {})
                attempt = int(record.get("attempt", 0)) + 1
                output = directory / "executions" / job["id"] / f"attempt-{attempt:03d}"
                output.parent.mkdir(parents=True, exist_ok=True)
                launcher_log = output.parent / f"attempt-{attempt:03d}.log"
                command = job_command(
                    study, job, directory, python=args.python, gpu=gpu, output=output
                )
                cpus = available_cpus[
                    lane * args.cpus_per_worker : (lane + 1) * args.cpus_per_worker
                ]
                if cpus:
                    command = ["taskset", "-c", ",".join(map(str, cpus)), *command]
                record = {
                    "status": "running",
                    "attempt": attempt,
                    "gpu": gpu,
                    "output": str(output),
                    "log": str(launcher_log),
                    "started_at": utc_now(),
                    "command": command,
                }
                status["jobs"][job["id"]] = record
                with launcher_log.open("x") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                record["pid"] = process.pid
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                print(
                    f"Started {job['id']} on GPU {gpu}; PID {process.pid}", flush=True
                )
            code = process.wait()
            with mutex:
                record["exit_code"] = code
                record["finished_at"] = utc_now()
                try:
                    if code:
                        raise RuntimeError(f"Worker exited {code}; see {launcher_log}")
                    record["verification"] = verify_job(study, job, output)
                    record["status"] = "completed"
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    stopped.set()
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                print(f"{job['id']}: {record['status']}", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, lane, gpu) for lane, gpu in enumerate(gpus)]
        for future in futures:
            try:
                future.result()
            except BaseException:
                stopped.set()
                status["phases"][args.phase] = "failed"
                with mutex:
                    write_json(status_path, status)
                raise
    complete = all(
        status["jobs"].get(job["id"], {}).get("status") == "completed" for job in jobs
    )
    status["phases"][args.phase] = "completed" if complete else "failed"
    status["updated_at"] = utc_now()
    write_json(status_path, status)
    lock_stream.close()
    if not complete:
        raise RuntimeError(
            "Phase is incomplete; successful jobs are retained for resume"
        )
    print(f"Phase {args.phase} completed", flush=True)


def show_status(args):
    directory = args.study_dir.expanduser().resolve()
    status = json.loads((directory / "status.json").read_text())
    rows = []
    for name, record in status["jobs"].items():
        ledger = Path(record["output"]) / "episodes.jsonl"
        count = sum(1 for _ in ledger.open()) if ledger.exists() else 0
        rows.append(
            {
                "job": name,
                "status": record["status"],
                "episodes": count,
                "gpu": record["gpu"],
                "pid": record.get("pid"),
            }
        )
    print(
        json.dumps(
            {"checked_at": utc_now(), "phases": status["phases"], "jobs": rows},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="Validate resources and create an immutable study"
    )
    plan.add_argument("--quality-run", type=Path, required=True)
    plan.add_argument("--policy-dir", type=Path, required=True)
    plan.add_argument("--hardware-profiles", type=Path, default=DEFAULT_ARCHIVE)
    plan.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    plan.add_argument("--episodes", type=int, default=512)
    plan.add_argument("--start-seed", type=int, default=0)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.set_defaults(handler=create_study)
    run = commands.add_parser("run", help="Run or resume a study phase")
    run.add_argument("--study-dir", type=Path, required=True)
    run.add_argument(
        "--phase", choices=("verification", "calibration", "validation"), required=True
    )
    run.add_argument(
        "--python",
        type=Path,
        required=True,
        help="Compatible KINETIX runtime interpreter",
    )
    run.add_argument(
        "--gpus",
        required=True,
        help="Explicit physical GPU indices; no automatic expansion",
    )
    run.add_argument("--cpus-per-worker", type=int, default=16)
    run.set_defaults(handler=run_phase)
    status = commands.add_parser(
        "status", help="Report progress without importing the model runtime"
    )
    status.add_argument("--study-dir", type=Path, required=True)
    status.set_defaults(handler=show_status)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
