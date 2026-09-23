"""Locate native state divergence and replay identical inputs without policy/RNG."""

import argparse
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_eval import benchmark_identity, state_digest  # noqa: E402


def named_arrays(value, prefix=""):
    import numpy as np

    if is_dataclass(value):
        children = [(field.name, getattr(value, field.name)) for field in fields(value)]
    elif isinstance(value, dict):
        children = list(value.items())
    elif isinstance(value, (list, tuple)):
        children = list(enumerate(value))
    else:
        return {prefix: np.asarray(value)}
    result = {}
    for name, child in children:
        result.update(named_arrays(child, f"{prefix}.{name}" if prefix else str(name)))
    return result


def restore_arrays(template, arrays, prefix=""):
    """Restore only matching array layouts into a known native state template."""
    import numpy as np

    def child(value, name):
        return restore_arrays(
            value, arrays, f"{prefix}.{name}" if prefix else str(name)
        )

    if is_dataclass(template):
        return replace(
            template,
            **{
                f.name: child(getattr(template, f.name), f.name)
                for f in fields(template)
            },
        )
    if isinstance(template, dict):
        return {key: child(value, key) for key, value in template.items()}
    if isinstance(template, (tuple, list)):
        return type(template)(
            child(value, index) for index, value in enumerate(template)
        )
    value, expected = np.asarray(arrays[prefix]), np.asarray(template)
    if value.shape != expected.shape or value.dtype != expected.dtype:
        raise ValueError(f"Different snapshot layout: {prefix}")
    return value


def compare_traces(left, right):
    return {
        "first_state_difference": next(
            (
                i + 1
                for i, (a, b) in enumerate(
                    zip(left["state_hashes"], right["state_hashes"])
                )
                if a != b
            ),
            None,
        ),
        "same_length": len(left["state_hashes"]) == len(right["state_hashes"]),
        "same_result": left["result"] == right["result"],
    }


def state_differences(left, right):
    import numpy as np

    a, b = named_arrays(left), named_arrays(right)
    if a.keys() != b.keys():
        raise ValueError("Different state layout")
    changes = []
    for key, x in a.items():
        y = b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"Different state layout: {key}")
        if x.tobytes() == y.tobytes():
            continue
        bits = np.frombuffer(x.tobytes(), np.uint8).reshape(x.shape + (x.itemsize,))
        other_bits = np.frombuffer(y.tobytes(), np.uint8).reshape(
            y.shape + (y.itemsize,)
        )
        mask = np.any(bits != other_bits, axis=-1)
        signed_zero = (
            int(np.sum((x == 0) & (y == 0) & (np.signbit(x) != np.signbit(y))))
            if x.dtype.kind == "f"
            else 0
        )
        changes.append(
            {
                "path": key,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "bitwise_different_elements": int(np.sum(mask)),
                "numerically_different_elements": int(np.sum(x != y)),
                "signed_zero_differences": signed_zero,
                "max_absolute_difference": float(
                    np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))
                ),
                "examples": [
                    {
                        "index": index.tolist(),
                        "reference": x[tuple(index)].item(),
                        "candidate": y[tuple(index)].item(),
                    }
                    for index in np.argwhere(mask)[:4]
                ],
            }
        )
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--level", default="grasp_easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--episode-repeats", type=int, default=10)
    parser.add_argument("--frozen-repeats", type=int, default=50)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument(
        "--frozen-input",
        type=Path,
        help="Replay a frozen_input.npz produced by this diagnostic, without model inference",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.episode_repeats < 2
        or args.frozen_repeats < 2
        or args.flow_steps < 1
        or args.gpu < 0
        or not 0 <= args.seed < 2**32
    ):
        parser.error(
            "Require two or more repeats, positive flow steps, nonnegative GPU and unsigned32 seed"
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
    identity["diagnostic_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (output / "diagnostic-source.py").write_bytes(Path(__file__).read_bytes())
    report = {
        "status": "running",
        "identity": identity,
        "level": args.level,
        "seed": args.seed,
        "flow_steps": args.flow_steps,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "episode_repeats": [],
        "first_divergence": None,
        "scope": "Diagnostic only: original reference rollouts and identical native engine inputs; no performance claim.",
    }

    def save():
        temp = output / "diagnostic.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(output / "diagnostic.json")

    save()
    env = KinetixEnvironment(args.level, action_noise_std=0.1)
    env.reset(args.seed)
    policy = (
        None
        if args.frozen_input
        else KinetixFlowPolicy(
            checkpoint, observation_dim=env.observation_dim, action_dim=env.action_dim
        )
    )
    original_step, original_process = env.step, env.native.action_type.process_action
    baseline, transition, frozen = None, None, None
    if args.frozen_input:
        path = args.frozen_input.expanduser().resolve()
        source_report = json.loads((path.parent / "diagnostic.json").read_text())
        if (
            source_report["level"] != args.level
            or source_report["identity"]["source_sha256"] != identity["source_sha256"]
        ):
            raise ValueError(
                "Frozen input uses a different level or native source identity"
            )
        with np.load(path, allow_pickle=False) as arrays:
            before = jax.tree.map(jax.numpy.asarray, restore_arrays(env.state, arrays))
            noisy_action = jax.numpy.asarray(arrays["noisy_action"])
            command = jax.numpy.asarray(arrays["processed_command"])
        if (
            state_digest((before, command))
            != source_report["frozen_replay"]["input_sha256"]
        ):
            raise ValueError(
                "Frozen input differs from its recorded state/command hash"
            )
        frozen = before, noisy_action, command
        transition = int(before.timestep)
        report["seed"] = source_report["seed"]
        report["flow_steps"] = source_report["flow_steps"]
        report["frozen_input_source"] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for repeat in range(0 if args.frozen_input else args.episode_repeats):
        states, inputs = [], []

        def process(action, state, static):
            command = original_process(action, state, static)
            # Retain immutable device arrays; add no synchronization before physics.
            inputs.append((state, action, command))
            return command

        def step(old, new, weights):
            result = original_step(old, new, weights)
            states.append(jax.device_get(env.state))
            return result

        env.step, env.native.action_type.process_action = step, process
        try:
            result = run_episode(
                policy,
                env,
                seed=args.seed,
                flow_steps=args.flow_steps,
                latency_ms=0,
                execute_horizon=4,
            )
        finally:
            env.step, env.native.action_type.process_action = (
                original_step,
                original_process,
            )
        hashes = [state_digest(state) for state in states]
        row = {"repeat": repeat, "result": result, "state_hashes": hashes}
        row["comparison_to_first"] = compare_traces(
            report["episode_repeats"][0] if report["episode_repeats"] else row, row
        )
        report["episode_repeats"].append(row)
        if baseline is None:
            baseline = (states, inputs, hashes)
        else:
            control = row["comparison_to_first"]["first_state_difference"]
            if control is not None and (
                report["first_divergence"] is None
                or control < report["first_divergence"]["control_step"]
            ):
                first = control - 1
                before, noisy_action, command = baseline[1][first]
                report["first_divergence"] = {
                    "repeat": repeat,
                    "control_step": first + 1,
                    "previous_state_equal": state_digest(before)
                    == state_digest(inputs[first][0]),
                    "noisy_action_equal": np.asarray(noisy_action).tobytes()
                    == np.asarray(inputs[first][1]).tobytes(),
                    "processed_command_equal": np.asarray(command).tobytes()
                    == np.asarray(inputs[first][2]).tobytes(),
                    "changed_leaves": state_differences(
                        baseline[0][first], states[first]
                    ),
                }
                transition, frozen = first, (before, noisy_action, command)
                np.savez_compressed(
                    output / "divergent_reference_state.npz",
                    **named_arrays(baseline[0][first]),
                )
                np.savez_compressed(
                    output / "divergent_repeat_state.npz", **named_arrays(states[first])
                )
        print(
            f"Reference repeat={repeat} controls={result['primitive_steps']} first_divergence={report['first_divergence'] and report['first_divergence']['control_step']}",
            flush=True,
        )
        save()
    if frozen is not None:
        before, noisy_action, command = frozen
        np.savez_compressed(
            output / "frozen_input.npz",
            **named_arrays(jax.device_get(before)),
            noisy_action=np.asarray(noisy_action),
            processed_command=np.asarray(command),
        )
        input_hash = state_digest((before, command))
        report["frozen_replay"] = {
            "control_step": transition + 1,
            "input_sha256": input_hash,
            "trials": [],
        }
        first_output, observed = None, set()
        for repeat in range(args.frozen_repeats):
            # This is the unchanged native compiled frame_skip=2 function, with
            # identical state and already-processed command; no model or RNG call.
            obs, state, reward, done, info = jax.device_get(
                env.native.engine_step(before, command, env.params)
            )
            digest = state_digest(state)
            if first_output is None:
                first_output = state
            if digest not in observed:
                np.savez_compressed(
                    output / f"frozen_output_{len(observed):02d}.npz",
                    **named_arrays(state),
                )
            observed.add(digest)
            report["frozen_replay"]["trials"].append(
                {
                    "repeat": repeat,
                    "state_sha256": digest,
                    "reward": float(reward),
                    "done": bool(done),
                    "solved": bool(info["GoalR"]),
                    "differences_from_first": state_differences(first_output, state),
                }
            )
        if state_digest((before, command)) != input_hash:
            raise RuntimeError("Frozen input mutated during replay")
        report["frozen_replay"]["unique_state_outputs"] = len(observed)
        if args.dump_hlo:
            lowered = type(env.native).engine_step.lower(
                env.native, before, command, env.params
            )
            (output / "native_engine_optimized_hlo.txt").write_text(
                lowered.compile().as_text()
            )
        print(
            f"Identical native engine inputs produced {len(observed)} state outputs over {args.frozen_repeats} repeats.",
            flush=True,
        )
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
