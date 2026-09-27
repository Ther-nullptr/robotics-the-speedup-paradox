"""Find bitwise differences in action preprocessing on a native reference rollout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_eval import (  # noqa: E402
    benchmark_identity,
    equivalent_trace,
    jit_action_processing,
    reference_episode,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--repeatability-trials", type=int, default=5)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.repeatability_trials < 2
        or args.flow_steps < 1
        or args.gpu < 0
        or not 0 <= args.seed < 2**32
    ):
        parser.error(
            "Use at least two repeats, positive flow steps, a nonnegative GPU and an unsigned32 seed"
        )
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    import jax
    import numpy as np
    from robotics_bench.kinetix.environment import KinetixEnvironment
    from robotics_bench.kinetix.policy import KinetixFlowPolicy
    from robotics_bench.kinetix.runner import run_episode

    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = (args.policy_dir / f"worlds_l_{args.level}.pkl").expanduser().resolve()
    identity = benchmark_identity(checkpoint, args.level, args.gpu)
    identity["diagnostic_source"] = str(Path(__file__).resolve())
    identity["diagnostic_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (output / "diagnostic-source.py").write_bytes(Path(__file__).read_bytes())
    env = KinetixEnvironment(args.level, action_noise_std=0.1)
    env.reset(args.seed)
    policy = KinetixFlowPolicy(
        checkpoint, observation_dim=env.observation_dim, action_dim=env.action_dim
    )
    reference = env.native.action_type.process_action
    jit_action_processing(env)
    candidate = env.native.action_type.process_action
    mismatches = []

    def compare(action, state, static):
        left = reference(action, state, static)
        right = candidate(action, state, static)
        a, b = (np.asarray(x) for x in jax.device_get((left, right)))
        if a.tobytes() != b.tobytes():
            mismatches.append(
                {
                    "control_step": env.control_steps + 1,
                    "input": np.asarray(jax.device_get(action)).tolist(),
                    "reference": a.tolist(),
                    "candidate": b.tolist(),
                    "reference_bits": a.view(np.uint32).tolist(),
                    "candidate_bits": b.view(np.uint32).tolist(),
                    "numerically_equal": bool(np.array_equal(a, b)),
                    "max_absolute_difference": float(np.max(np.abs(a - b))),
                }
            )
        return left

    env.native.action_type.process_action = compare
    result = run_episode(
        policy,
        env,
        seed=args.seed,
        flow_steps=args.flow_steps,
        latency_ms=0,
        execute_horizon=4,
    )
    env.native.action_type.process_action = reference
    reference_repeats = [
        reference_episode(
            policy, env, seed=args.seed, flow_steps=args.flow_steps, capture=True
        )
        for _ in range(args.repeatability_trials)
    ]
    env.native.action_type.process_action = candidate
    candidate_repeats = [
        reference_episode(
            policy, env, seed=args.seed, flow_steps=args.flow_steps, capture=True
        )
        for _ in range(args.repeatability_trials)
    ]
    report = {
        "level": args.level,
        "seed": args.seed,
        "flow_steps": args.flow_steps,
        "scope": "Action preprocessing only, identical input/state; reference drives the rollout. No timing claim.",
        "identity": identity,
        "result": result,
        "command_mismatches": mismatches,
        "repeatability_trials": args.repeatability_trials,
        "reference_self_equivalent": all(
            equivalent_trace(reference_repeats[0], r) for r in reference_repeats[1:]
        ),
        "candidate_self_equivalent": all(
            equivalent_trace(candidate_repeats[0], r) for r in candidate_repeats[1:]
        ),
        "reference_repeats": reference_repeats,
        "candidate_repeats": candidate_repeats,
    }
    (output / "diagnostic.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Compared {result['primitive_steps']} controls; preprocessing mismatches: {len(mismatches)}"
    )
    print(
        f"Reference repeatable: {report['reference_self_equivalent']}; candidate repeatable: {report['candidate_self_equivalent']}"
    )


if __name__ == "__main__":
    main()
