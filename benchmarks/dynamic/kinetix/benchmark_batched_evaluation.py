"""Validate per-seed trajectories before measuring fixed-workload throughput."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batched_evaluation import BatchedEvaluator, episode_index, seed_groups  # noqa: E402
from benchmark_eval import benchmark_identity, equivalent_trace  # noqa: E402
from resident_action_runner import metadata_matches  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--level", default="car_launch")
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--batch-sizes", default="8,16,32")
    parser.add_argument(
        "--method", choices=("vmap", "threads", "threads-locked"), default="vmap"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(value) for value in args.batch_sizes.split(",")]
    seeds = list(range(args.start_seed, args.start_seed + args.episodes))
    if (
        args.gpu < 0
        or args.flow_steps < 1
        or args.repeats < 1
        or len(set(sizes)) != len(sizes)
    ):
        parser.error("Invalid GPU, flow count, repeats or batch sizes")
    for size in sizes:
        seed_groups(seeds, size)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    import jax

    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    identity = benchmark_identity(
        (args.policy_dir / f"worlds_l_{args.level}.pkl").expanduser().resolve(),
        args.level,
        args.gpu,
    )
    for name in (
        "benchmark_batched_evaluation.py",
        "batched_evaluation.py",
        "resident_action_runner.py",
        "device_chunk_candidate.py",
    ):
        source = Path(__file__).with_name(name).read_bytes()
        identity["source_sha256"][name] = hashlib.sha256(source).hexdigest()
        (output / name).write_bytes(source)
    report = {
        "format": "kinetix-batched-evaluation-v1",
        "status": "running",
        "identity": identity,
        "level": args.level,
        "seeds": seeds,
        "flow_steps": args.flow_steps,
        "method": args.method,
        "batch_sizes": sizes,
        "repeats": args.repeats,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "model_batch_size": 1,
        "traces": {},
        "checks": {},
        "timings": {},
        "validated_speedups": {},
        "scope": "Fixed same-task seed workload; makespan includes reset, packing and result collection. Excludes model load, first compilation, trace and file output.",
        "memory_scope": "JAX allocator process-cumulative peak after each phase, including cached executables; not a per-batch isolated allocation peak.",
        "cold_capture_scope": "First candidate workload includes compilation and full trace capture; not compilation-only time.",
    }

    def save():
        tmp = output / "benchmark.json.tmp"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(output / "benchmark.json")

    def memory():
        return {
            "jax_allocator": jax.devices()[0].memory_stats(),
            "nvidia_gpu_fields": "memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "nvidia_gpu": subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(args.gpu),
                    "--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip(),
        }

    save()
    try:
        evaluate(args, report, save, memory)
    except BaseException as error:
        report["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise


def evaluate(args, report, save, memory):
    from robotics_bench.kinetix.environment import KinetixEnvironment
    from robotics_bench.kinetix.policy import KinetixFlowPolicy

    identity = report["identity"]
    seeds, sizes = report["seeds"], report["batch_sizes"]
    setup_start = time.perf_counter()
    env = KinetixEnvironment(args.level, action_noise_std=0.1)
    env.reset(seeds[0])
    policy = KinetixFlowPolicy(
        (args.policy_dir / f"worlds_l_{args.level}.pkl").expanduser().resolve(),
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
    )
    evaluator = BatchedEvaluator(policy, env, args.flow_steps)
    report.update(
        protocol=env.metadata,
        policy_metadata=policy.metadata,
        setup_seconds=time.perf_counter() - setup_start,
    )
    start = time.perf_counter()
    evaluator.serial(seeds[:1])
    report["serial_first_episode_warmup_seconds"] = time.perf_counter() - start
    print("Capturing serial reference trajectories", flush=True)
    baseline = evaluator.serial(seeds, capture=True)
    print("Capturing serial repeat trajectories", flush=True)
    repeated = evaluator.serial(seeds, capture=True)
    reference = episode_index(baseline["episodes"], seeds)
    repeated_index = episode_index(repeated["episodes"], seeds)
    report["traces"].update(serial=baseline, serial_repeat=repeated)
    report["serial_repeatable"] = {
        str(seed): equivalent_trace(reference[seed], repeated_index[seed])
        for seed in seeds
    }
    if args.reference_run:
        prior = json.loads(args.reference_run.read_text())
        if (
            prior["level"] != args.level
            or prior["flow_steps"] != args.flow_steps
            or prior["identity"]["checkpoint"]["sha256"]
            != identity["checkpoint"]["sha256"]
            or not metadata_matches(prior["protocol"], env.metadata)
            or not metadata_matches(prior["policy_metadata"], policy.metadata)
            or prior.get("xla_flags", "") != report["xla_flags"]
        ):
            raise ValueError("Historical reference task/model/protocol/mode mismatch")
        common = sorted(set(seeds) & {int(seed) for seed in prior["traces"]})
        if not common:
            raise ValueError("Historical reference has no overlapping seeds to verify")
        report["historical_reference"] = {
            "path": str(args.reference_run),
            "sha256": hashlib.sha256(args.reference_run.read_bytes()).hexdigest(),
            "checked_seeds": common,
            "unavailable_seeds": sorted(set(seeds) - set(common)),
            "equivalent": {
                str(seed): equivalent_trace(
                    reference[seed], prior["traces"][str(seed)]["resident"]
                )
                for seed in common
            },
        }
    save()
    baseline_valid = all(report["serial_repeatable"].values()) and all(
        report.get("historical_reference", {}).get("equivalent", {}).values()
    )
    report["baseline_valid"] = baseline_valid
    print(f"Serial reference exact and repeatable: {baseline_valid}", flush=True)
    for size in sizes:

        def call(capture=False):
            return (
                evaluator.vectorized(seeds, size, capture=capture)
                if args.method == "vmap"
                else evaluator.threaded(
                    seeds,
                    size,
                    capture=capture,
                    serialize_windows=args.method == "threads-locked",
                )
            )

        print(f"Checking {args.method} width={size}", flush=True)
        warm = time.perf_counter()
        trial = call(capture=True)
        report.setdefault("first_candidate_capture_seconds", {})[str(size)] = (
            time.perf_counter() - warm
        )
        again = call(capture=True)
        candidate = episode_index(trial["episodes"], seeds)
        second = episode_index(again["episodes"], seeds)
        checks = {
            str(seed): {
                "equivalent": equivalent_trace(reference[seed], candidate[seed]),
                "repeatable": equivalent_trace(candidate[seed], second[seed]),
                "same_result": reference[seed]["result"] == candidate[seed]["result"],
                "first_state_difference": next(
                    (
                        i + 1
                        for i, (a, b) in enumerate(
                            zip(
                                reference[seed]["state_hashes"],
                                candidate[seed]["state_hashes"],
                            )
                        )
                        if a != b
                    ),
                    None,
                ),
            }
            for seed in seeds
        }
        report["traces"][str(size)] = {"candidate": trial, "repeat": again}
        report["checks"][str(size)] = checks
        report.setdefault("memory", {})[str(size)] = memory()
        passed = baseline_valid and all(
            row["equivalent"] and row["repeatable"] for row in checks.values()
        )
        print(
            f"width={size}: exact={passed}, differing_seeds={[seed for seed, row in checks.items() if not row['equivalent']]}",
            flush=True,
        )
        report["validated_speedups"][str(size)] = None
        save()
        if not passed or args.gate_only:
            continue
        timings = []
        for repeat in range(args.repeats):
            for mode in (
                ["serial", "parallel"] if repeat % 2 == 0 else ["parallel", "serial"]
            ):
                result = evaluator.serial(seeds) if mode == "serial" else call()
                rows = episode_index(result["episodes"], seeds)
                if any(
                    rows[seed]["result"] != reference[seed]["result"] for seed in seeds
                ):
                    raise RuntimeError(
                        "Timed outcomes differ from the verified workload"
                    )
                result.update(mode=mode, repeat=repeat)
                timings.append(result)
                print(
                    f"width={size} repeat={repeat} mode={mode} episodes={len(seeds)} wall={result['wall_seconds']:.3f}s",
                    flush=True,
                )
            report["timings"][str(size)] = timings
            save()
        totals = {
            mode: sum(row["wall_seconds"] for row in timings if row["mode"] == mode)
            for mode in ("serial", "parallel")
        }
        report["validated_speedups"][str(size)] = totals["serial"] / totals["parallel"]
        report.setdefault("throughput", {})[str(size)] = {
            mode: len(seeds) * args.repeats / total for mode, total in totals.items()
        }
        save()
    report["status"] = "completed"
    save()
    print(json.dumps(report["validated_speedups"]), flush=True)


if __name__ == "__main__":
    main()
