"""Compare host-roundtrip and GPU-resident actions with matched native controls."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_eval import (
    benchmark_identity,
    equivalent_trace,
    reference_episode,
    state_digest,
)  # noqa: E402
from device_chunk_candidate import compile_device_window, device_chunks  # noqa: E402
from resident_action_runner import metadata_matches, resident_episode  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--level", default="car_launch")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--reference-run",
        type=Path,
        help="Earlier device-chunk benchmark.json for historical native-trace validation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if (
        args.flow_steps < 1
        or args.repeats < 1
        or args.gpu < 0
        or not seeds
        or len(seeds) != len(set(seeds))
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
    import jax
    import jax.numpy as jnp
    from robotics_bench.kinetix.environment import KinetixEnvironment
    from robotics_bench.kinetix.policy import KinetixFlowPolicy

    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = (args.policy_dir / f"worlds_l_{args.level}.pkl").expanduser().resolve()
    identity = benchmark_identity(checkpoint, args.level, args.gpu)
    for name in (
        "benchmark_resident_actions.py",
        "resident_action_runner.py",
        "device_chunk_candidate.py",
    ):
        source = Path(__file__).with_name(name).read_bytes()
        identity["source_sha256"][name] = hashlib.sha256(source).hexdigest()
        (output / name).write_bytes(source)
    report = {
        "format": "kinetix-resident-actions-v1",
        "status": "running",
        "identity": identity,
        "level": args.level,
        "seeds": seeds,
        "flow_steps": args.flow_steps,
        "repeats": args.repeats,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "traces": {},
        "checks": {},
        "rows": [],
        "validated_incremental_speedup": None,
        "timing_scope": "Warm episode wall time including reset, excluding trace capture, file output and first compilation. Baseline: host-roundtrip device chunks.",
        "policy_timing": "Resident policy_enqueue_seconds measures host submission only; GPU work completes before the device-window metadata wait returns.",
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
    report.update(protocol=env.metadata, policy_metadata=policy.metadata)
    host_context = device_chunks(policy, env)
    advance = compile_device_window(env, resident_actions=True)
    prior = None
    if args.reference_run:
        prior = json.loads(args.reference_run.read_text())
        if (
            prior["level"] != args.level
            or prior["flow_steps"] != args.flow_steps
            or prior["identity"]["checkpoint"]["sha256"]
            != identity["checkpoint"]["sha256"]
            or prior["xla_flags"] != report["xla_flags"]
            or not metadata_matches(prior["protocol"], env.metadata)
            or not metadata_matches(prior["policy_metadata"], policy.metadata)
            or any(str(seed) not in prior["traces"] for seed in seeds)
        ):
            report.update(
                status="failed", error="Historical reference metadata mismatch"
            )
            save()
            raise ValueError(
                "Historical reference does not match the task, checkpoint, model, physics, N, seeds or execution mode"
            )
        report["reference_run"] = {
            "path": str(args.reference_run),
            "sha256": hashlib.sha256(args.reference_run.read_bytes()).hexdigest(),
        }

    # The unused prediction tail must be validated before any physical advance.
    before_hash = state_digest(env.state)
    report["invalid_action_guards"] = {}
    for name, index, value in (
        ("nan_in_unused_tail", -1, jnp.nan),
        ("infinity_in_executed_prefix", 0, jnp.inf),
    ):
        bad_actions = (
            jnp.zeros((policy.action_horizon, env.action_dim), dtype=jnp.float32)
            .at[index, 0]
            .set(value)
        )
        stopped_states, _, _, count, valid = advance(
            env.state, env._noise_rng, jnp.int32(0), bad_actions, env.params
        )
        count, valid = jax.device_get((count, valid))
        report["invalid_action_guards"][name] = (
            not bool(valid)
            and int(count) == 0
            and all(state_digest(s) == before_hash for s in stopped_states)
        )
    if not all(report["invalid_action_guards"].values()):
        raise RuntimeError("Invalid model actions advanced the simulator")

    def episode(mode, seed, capture):
        if mode == "resident":
            return resident_episode(
                policy,
                env,
                advance,
                seed=seed,
                flow_steps=args.flow_steps,
                capture=capture,
            )
        if mode == "host_chunks":
            with host_context():
                return reference_episode(
                    policy, env, seed=seed, flow_steps=args.flow_steps, capture=capture
                )
        return reference_episode(
            policy, env, seed=seed, flow_steps=args.flow_steps, capture=capture
        )

    for seed in seeds:
        traces = {}
        for mode in (
            "native",
            "native_repeat",
            "host_chunks",
            "resident",
            "resident_repeat",
        ):
            print(f"Checking seed={seed} mode={mode}", flush=True)
            traces[mode] = episode(
                "resident" if mode == "resident_repeat" else mode, seed, True
            )
        report["traces"][str(seed)] = traces
        checks = {
            "native_repeatable": equivalent_trace(
                traces["native"], traces["native_repeat"]
            ),
            "host_chunks_equivalent": equivalent_trace(
                traces["native"], traces["host_chunks"]
            ),
            "resident_equivalent": equivalent_trace(
                traces["native"], traces["resident"]
            ),
            "resident_repeatable": equivalent_trace(
                traces["resident"], traces["resident_repeat"]
            ),
        }
        if prior:
            checks["historical_native_equivalent"] = equivalent_trace(
                prior["traces"][str(seed)]["reference"], traces["native"]
            )
        report["checks"][str(seed)] = checks
        print(f"seed={seed} checks={checks}", flush=True)
        save()
    if not all(all(check.values()) for check in report["checks"].values()):
        report["status"] = "equivalence_failed"
        save()
        print("Timing withheld: trajectory gate failed.", flush=True)
        return
    for repeat in range(args.repeats):
        for seed in seeds:
            for mode in (
                ["host_chunks", "resident"]
                if repeat % 2 == 0
                else ["resident", "host_chunks"]
            ):
                row = episode(mode, seed, False)
                row.update(mode=mode, repeat=repeat, seed=seed)
                report["rows"].append(row)
                print(
                    f"{mode} repeat={repeat} seed={seed} wall={row['wall_seconds']:.4f}s",
                    flush=True,
                )
        save()
    valid_outcomes = all(
        row["result"] == report["traces"][str(row["seed"])]["native"]["result"]
        for row in report["rows"]
    )
    resident_reads = sum(
        r["full_action_host_reads"] for r in report["rows"] if r["mode"] == "resident"
    )
    totals = {
        mode: sum(r["wall_seconds"] for r in report["rows"] if r["mode"] == mode)
        for mode in ("host_chunks", "resident")
    }
    valid_timing = all(
        math.isfinite(r["wall_seconds"]) and r["wall_seconds"] > 0
        for r in report["rows"]
    )
    passed = valid_outcomes and resident_reads == 0 and valid_timing
    ratios = []
    for repeat in range(args.repeats):
        pair = {
            mode: sum(
                r["wall_seconds"]
                for r in report["rows"]
                if r["mode"] == mode and r["repeat"] == repeat
            )
            for mode in totals
        }
        ratios.append(pair["host_chunks"] / pair["resident"])
    report.update(
        status="completed",
        timed_outcomes_equal=valid_outcomes,
        resident_timed_action_host_reads=resident_reads,
        rollout_seconds=totals,
        paired_repeat_speedups=ratios if passed else [],
        validated_incremental_speedup=totals["host_chunks"] / totals["resident"]
        if passed
        else None,
    )
    save()
    print(
        json.dumps(
            {
                "rollout_seconds": totals,
                "validated_incremental_speedup": report[
                    "validated_incremental_speedup"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
