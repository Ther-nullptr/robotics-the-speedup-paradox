"""Measure evaluation wall time while retaining the native simulation contract."""

import argparse
from collections import defaultdict
import cProfile
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def benchmark_identity(checkpoint, level, gpu):
    """Collect provenance outside measured episodes, including uncommitted code."""
    source = ROOT / "src/robotics_bench/kinetix"
    files = [Path(__file__), source / "levels" / f"{level}.json"]
    files.extend(sorted(source.rglob("*.py")))

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def command(args):
        return subprocess.check_output(args, text=True, timeout=15).strip()

    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "repository_status": command(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "checkpoint": {"path": str(checkpoint), "sha256": digest(checkpoint)},
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "jax",
                "jaxlib",
                "flax",
                "numpy",
                "jaxgl",
                "gymnax",
                "jaxued",
                "chex",
            )
        },
        "gpu_query_fields": "index,uuid,name,driver_version,memory.total",
        "gpu_query": command(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def state_digest(state):
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(jax.device_get(state)):
        value = np.asarray(leaf)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def jit_action_processing(env):
    """Compile only native gather/clip/select; retain the physics JIT boundary."""
    import jax

    original = env.native.action_type.process_action

    def prepare(action, motor_bindings, thruster_bindings, motor_auto):
        metadata = SimpleNamespace(
            motor_bindings=motor_bindings,
            thruster_bindings=thruster_bindings,
            motor_auto=motor_auto,
        )
        return original(action, metadata, env.static)

    compiled = jax.jit(prepare)

    def process(action, state, static):
        if static is not env.static:
            raise ValueError(
                "Action processing received an unexpected static configuration"
            )
        return compiled(
            action, state.motor_bindings, state.thruster_bindings, state.motor_auto
        )

    env.native.action_type.process_action = process


def reference_episode(policy, env, *, seed, flow_steps, capture=False):
    from robotics_bench.kinetix.runner import run_episode

    spans, states, actions, events = defaultdict(float), [], [], []
    original_step, original_infer = env.step, policy.infer

    def step(old, new, weights):
        start = time.perf_counter()
        output = original_step(old, new, weights)
        spans["environment_step_seconds"] += time.perf_counter() - start
        if capture:
            states.append(state_digest(env.state))
        return output

    def infer(obs, count):
        start = time.perf_counter()
        output = original_infer(obs, count)
        spans["policy_call_seconds"] += time.perf_counter() - start
        if capture:
            actions.append(hashlib.sha256(output.tobytes()).hexdigest())
        return output

    env.step, policy.infer = step, infer
    start = time.perf_counter()
    try:
        result = run_episode(
            policy,
            env,
            seed=seed,
            flow_steps=flow_steps,
            latency_ms=0,
            execute_horizon=4,
            on_event=events.append if capture else None,
        )
    finally:
        env.step, policy.infer = original_step, original_infer
    wall = time.perf_counter() - start
    return {
        "result": result,
        "wall_seconds": wall,
        "components": dict(spans),
        "other_seconds": wall - sum(spans.values()),
        "state_hashes": states,
        "action_hashes": actions,
        "events": events,
    }


def equivalent_trace(left, right):
    """Outcome equality alone is insufficient for claiming unchanged evaluation."""
    for trace in (left, right):
        steps = trace["result"]["primitive_steps"]
        calls = trace["result"]["inference_calls"]
        if (
            steps < 1
            or calls < 1
            or len(trace["state_hashes"]) != steps
            or len(trace["action_hashes"]) != calls
            or sum(event["kind"] == "control" for event in trace["events"]) != steps
            or sum(event["kind"] == "inference" for event in trace["events"]) != calls
        ):
            return False
    if any(
        left[key] != right[key] for key in ("result", "state_hashes", "action_hashes")
    ):
        return False

    def semantic_events(trace):
        return [
            {k: v for k, v in row.items() if k != "host_policy_call_seconds"}
            for row in trace["events"]
        ]

    return semantic_events(left) == semantic_events(right)


def comparison_summary(report):
    """Validate both timed outcomes and separate detailed traces before speedup."""
    seeds, repeats = report["seeds"], report["repeats"]
    traces = report["equivalence_traces"]
    reference, candidate = traces["reference"], traces["preprocess_jit"]
    exact = {
        str(seed): equivalent_trace(a, b)
        for seed, a, b in zip(seeds, reference, candidate, strict=True)
    }
    grouped = defaultdict(list)
    for row in report["rows"]:
        grouped[(row["repeat"], row["seed"], row["mode"])].append(row)
    outcomes, totals = {}, defaultdict(float)
    for repeat in range(repeats):
        for index, seed in enumerate(seeds):
            left = grouped[(repeat, seed, "reference")]
            right = grouped[(repeat, seed, "preprocess_jit")]
            outcomes[f"{repeat}:{seed}"] = (
                len(left) == len(right) == 1
                and left[0]["result"]
                == right[0]["result"]
                == reference[index]["result"]
            )
            for mode, rows in (("reference", left), ("preprocess_jit", right)):
                totals[(repeat, mode)] += sum(row["wall_seconds"] for row in rows)
    valid = (
        bool(outcomes)
        and len(report["rows"]) == 2 * len(seeds) * repeats
        and all(exact.values())
        and all(outcomes.values())
        and all(value > 0 for value in totals.values())
    )
    return {
        "exact_equivalence_by_seed": exact,
        "timed_outcomes_equal_by_pair": outcomes,
        "validated_speedup": (
            sum(totals[(i, "reference")] for i in range(repeats))
            / sum(totals[(i, "preprocess_jit")] for i in range(repeats))
            if valid
            else None
        ),
        "paired_repeat_speedups": (
            [
                totals[(i, "reference")] / totals[(i, "preprocess_jit")]
                for i in range(repeats)
            ]
            if valid
            else []
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-dir", type=Path, required=True)
    p.add_argument("--level", default="car_launch")
    p.add_argument("--flow-steps", type=int, default=5)
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gpu", type=int, required=True)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument(
        "--preprocess-jit",
        action="store_true",
        help="JIT the original action gather/clip/select separately from physics",
    )
    modes.add_argument(
        "--compare-preprocess",
        action="store_true",
        help="Alternate reference/candidate in one process and require exact full-episode traces",
    )
    p.add_argument(
        "--capture-traces",
        action="store_true",
        help="Record exact action and full-state hashes after timing for equivalence checks",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.flow_steps < 1 or args.repeats < 1:
        p.error("flow-steps and repeats must be positive")
    if args.gpu < 0:
        p.error("gpu must be nonnegative")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not 0 <= s < 2**32 for s in seeds)
    ):
        p.error("seeds must be distinct unsigned32 values")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONDONTWRITEBYTECODE="1",
        SDL_VIDEODRIVER="dummy",
    )
    started = time.perf_counter()
    from robotics_bench.kinetix.environment import KinetixEnvironment
    from robotics_bench.kinetix.policy import KinetixFlowPolicy

    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = (args.policy_dir / f"worlds_l_{args.level}.pkl").expanduser().resolve()
    identity = benchmark_identity(checkpoint, args.level, args.gpu)
    env = KinetixEnvironment(args.level, action_noise_std=0.1)
    env.reset(seeds[0])
    processors = {"reference": env.native.action_type.process_action}
    if args.preprocess_jit or args.compare_preprocess:
        jit_action_processing(env)
        processors["preprocess_jit"] = env.native.action_type.process_action
    mode_names = (
        ["reference", "preprocess_jit"]
        if args.compare_preprocess
        else ["preprocess_jit" if args.preprocess_jit else "reference"]
    )
    policy = KinetixFlowPolicy(
        checkpoint,
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
    )
    report = {
        "format": "kinetix-eval-benchmark-v1",
        "status": "running",
        "identity": identity,
        "level": args.level,
        "flow_steps": args.flow_steps,
        "seeds": seeds,
        "repeats": args.repeats,
        "scope": "native_batch1_runner_without_file_logging_or_video",
        "initialization_seconds": time.perf_counter() - started,
        "rows": [],
        "protocol": env.metadata,
        "gpu": args.gpu,
        "preprocess_jit": args.preprocess_jit,
        "compare_preprocess": args.compare_preprocess,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "policy_metadata": policy.metadata,
    }
    report["warmups"] = {}
    for mode in mode_names:
        env.native.action_type.process_action = processors[mode]
        report["warmups"][mode] = reference_episode(
            policy, env, seed=seeds[0], flow_steps=args.flow_steps
        )
        print(
            f"{mode} warmup: {report['warmups'][mode]['wall_seconds']:.3f}s", flush=True
        )
    for repeat in range(args.repeats):
        for seed in seeds:
            for mode in mode_names if repeat % 2 == 0 else list(reversed(mode_names)):
                env.native.action_type.process_action = processors[mode]
                row = reference_episode(
                    policy, env, seed=seed, flow_steps=args.flow_steps
                )
                row.update(mode=mode, repeat=repeat, seed=seed)
                report["rows"].append(row)
                print(
                    f"{mode} repeat={repeat} seed={seed} controls={row['result']['primitive_steps']} wall={row['wall_seconds']:.3f}s",
                    flush=True,
                )
    report["summary"] = {}
    for mode in mode_names:
        selected = [r for r in report["rows"] if r["mode"] == mode]
        walls = [r["wall_seconds"] for r in selected]
        report["summary"][mode] = {
            "mean_episode_seconds": statistics.mean(walls),
            "episodes_per_second": len(walls) / sum(walls),
            "environment_seconds": sum(
                r["components"]["environment_step_seconds"] for r in selected
            ),
            "policy_seconds": sum(
                r["components"]["policy_call_seconds"] for r in selected
            ),
            "rollout_seconds": sum(walls),
        }
    if args.capture_traces or args.compare_preprocess:
        report["equivalence_traces"] = {}
        for mode in mode_names:
            env.native.action_type.process_action = processors[mode]
            report["equivalence_traces"][mode] = [
                reference_episode(
                    policy, env, seed=seed, flow_steps=args.flow_steps, capture=True
                )
                for seed in seeds
            ]
        print("Exact episode traces recorded outside the timing samples.", flush=True)
    if args.compare_preprocess:
        report.update(comparison_summary(report))
    profile_mode = "reference" if "reference" in mode_names else "preprocess_jit"
    env.native.action_type.process_action = processors[profile_mode]
    profile = cProfile.Profile()
    profile.enable()
    reference_episode(policy, env, seed=seeds[0], flow_steps=args.flow_steps)
    profile.disable()
    profile.dump_stats(output / f"{profile_mode}.pstats")
    with (output / f"{profile_mode}_profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(35)
    report["status"] = "completed"
    (output / "benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
