"""Gate experimental device chunks against unchanged native reference episodes."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_eval import benchmark_identity, equivalent_trace, reference_episode  # noqa: E402
from device_chunk_candidate import device_chunks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--level", default="car_launch")
    parser.add_argument("--seeds", default="0,2")
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if (
        args.flow_steps < 1
        or args.repeats < 1
        or args.gpu < 0
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(not 0 <= x < 2**32 for x in seeds)
    ):
        parser.error(
            "Use distinct unsigned32 seeds, positive counts and a nonnegative GPU"
        )
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    from robotics_bench.kinetix.environment import KinetixEnvironment
    from robotics_bench.kinetix.policy import KinetixFlowPolicy

    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = (args.policy_dir / f"worlds_l_{args.level}.pkl").expanduser().resolve()
    identity = benchmark_identity(checkpoint, args.level, args.gpu)
    for file in (Path(__file__), Path(__file__).with_name("device_chunk_candidate.py")):
        identity["source_sha256"][str(file.name)] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
        (output / file.name).write_bytes(file.read_bytes())
    report = {
        "status": "running",
        "identity": identity,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "level": args.level,
        "seeds": seeds,
        "flow_steps": args.flow_steps,
        "repeats": args.repeats,
        "scope": "Zero-delay native execute-four device-chunk experiment; native budget, per-control termination/finite checks.",
        "traces": {},
        "rows": [],
        "validated_speedup": None,
    }

    def save():
        temp = output / "benchmark.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(output / "benchmark.json")

    save()
    env = KinetixEnvironment(args.level, action_noise_std=0.1)
    env.reset(seeds[0])
    policy = KinetixFlowPolicy(
        checkpoint, observation_dim=env.observation_dim, action_dim=env.action_dim
    )
    report["protocol"] = env.metadata
    report["policy_metadata"] = policy.metadata
    contexts = {"reference": nullcontext, "device_chunks": device_chunks(policy, env)}
    checks = {}
    for seed in seeds:
        traces = {}
        for mode in ("reference", "reference_repeat", "device_chunks"):
            with contexts["reference" if mode == "reference_repeat" else mode]():
                traces[mode] = reference_episode(
                    policy, env, seed=seed, flow_steps=args.flow_steps, capture=True
                )
        report["traces"][str(seed)] = traces
        checks[str(seed)] = {
            "reference_repeatable": equivalent_trace(
                traces["reference"], traces["reference_repeat"]
            ),
            "candidate_equivalent": equivalent_trace(
                traces["reference"], traces["device_chunks"]
            ),
            "same_result": traces["reference"]["result"]
            == traces["device_chunks"]["result"],
            "first_state_difference": next(
                (
                    i + 1
                    for i, (a, b) in enumerate(
                        zip(
                            traces["reference"]["state_hashes"],
                            traces["device_chunks"]["state_hashes"],
                        )
                    )
                    if a != b
                ),
                None,
            ),
        }
        print(f"seed={seed} checks={checks[str(seed)]}", flush=True)
        save()
    report["checks"] = checks
    if not all(
        row["reference_repeatable"] and row["candidate_equivalent"]
        for row in checks.values()
    ):
        report["status"] = "equivalence_failed"
        save()
        print("Timing withheld: exact trajectory gate failed.", flush=True)
        return
    for repeat in range(args.repeats):
        for seed in seeds:
            for mode in (
                ["reference", "device_chunks"]
                if repeat % 2 == 0
                else ["device_chunks", "reference"]
            ):
                with contexts[mode]():
                    row = reference_episode(
                        policy, env, seed=seed, flow_steps=args.flow_steps
                    )
                row.update(mode=mode, repeat=repeat, seed=seed)
                report["rows"].append(row)
                print(
                    f"{mode} repeat={repeat} seed={seed} wall={row['wall_seconds']:.3f}s",
                    flush=True,
                )
    outcome_match = all(
        row["result"] == report["traces"][str(row["seed"])]["reference"]["result"]
        for row in report["rows"]
    )
    totals = {
        mode: sum(row["wall_seconds"] for row in report["rows"] if row["mode"] == mode)
        for mode in contexts
    }
    report.update(
        status="completed",
        timed_outcomes_equal=outcome_match,
        rollout_seconds=totals,
        validated_speedup=totals["reference"] / totals["device_chunks"]
        if outcome_match
        else None,
    )
    save()
    print(
        json.dumps(
            {
                "rollout_seconds": totals,
                "validated_speedup": report["validated_speedup"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
