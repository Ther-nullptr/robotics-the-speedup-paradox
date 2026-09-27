"""Zero-delay evaluation with separate model/physics JITs and GPU-resident actions."""

import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_eval import state_digest  # noqa: E402
from robotics_bench.kinetix.protocol import delay_plan  # noqa: E402


def metadata_matches(stored, current):
    def canonical(value):
        return json.dumps(
            {k: v for k, v in value.items() if k != "source_file"},
            sort_keys=True,
            allow_nan=False,
        )

    return canonical(stored) == canonical(current)


def validate_action_layout(actions, *, horizon, action_dim):
    # Metadata only: materializing values here would restore the host round trip.
    if actions.shape != (horizon, action_dim) or actions.dtype.kind != "f":
        raise ValueError(
            "Policy must return a floating-point action chunk of the declared shape"
        )


def resident_episode(policy, env, advance, *, seed, flow_steps, capture=False):
    """Validate actions on GPU before any physics; transfer only window metadata.

    Full action values reach the host only for explicitly requested trace capture,
    which the benchmark excludes from timing. Enqueue time is not policy latency.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    states, actions, events = [], [], []
    spans = {"policy_enqueue_seconds": 0.0, "window_call_and_wait_seconds": 0.0}
    # Match the reference wrapper: validation and plan construction are timed.
    start = time.perf_counter()
    if (
        type(seed) is not int
        or type(flow_steps) is not int
        or not 0 <= seed < 2**32
        or flow_steps < 1
        or policy.action_horizon != 8
    ):
        raise ValueError(
            "Resident evaluation requires unsigned32 seed, positive N and horizon eight"
        )
    if env.max_steps != int(env.params.max_timesteps):
        raise ValueError("Resident evaluation requires the native control budget")
    plan = delay_plan(
        0,
        physics_dt=env.physics_dt,
        frame_skip=env.frame_skip,
        execute_horizon=4,
        action_horizon=policy.action_horizon,
    )
    observation = env.reset(seed)
    initial_hash = hashlib.sha256(np.asarray(observation).tobytes()).hexdigest()
    policy.reset(seed)
    steps = calls = host_action_reads = 0
    total_reward = 0.0
    success, reason, previous_source_step = False, "budget_exhausted", None
    while steps < env.max_steps:
        source_step = steps
        enqueue_start = time.perf_counter()
        chunk = policy.infer_device(observation, flow_steps)
        spans["policy_enqueue_seconds"] += time.perf_counter() - enqueue_start
        validate_action_layout(
            chunk, horizon=policy.action_horizon, action_dim=env.action_dim
        )
        if not isinstance(chunk, jax.Array) or not all(
            d.platform == "gpu" for d in chunk.devices()
        ):
            raise ValueError("Resident policy output must be a GPU JAX array")
        window_start = time.perf_counter()
        window_states, observations, values, count, valid = advance(
            env.state,
            env._noise_rng,
            jnp.int32(steps),
            chunk,
            env.params,
        )
        values, count, valid = jax.device_get((values, count, valid))
        spans["window_call_and_wait_seconds"] += time.perf_counter() - window_start
        if not bool(valid):
            raise ValueError(
                "Policy must return a finite floating-point action chunk of the declared shape"
            )
        if not 1 <= int(count) <= min(4, env.max_steps - steps):
            raise RuntimeError("Invalid executed control count from device window")
        calls += 1
        if capture:
            host_chunk = np.asarray(jax.device_get(chunk))
            host_action_reads += 1
            digest = hashlib.sha256(host_chunk.tobytes()).hexdigest()
            actions.append(digest)
            events.append(
                {
                    "kind": "inference",
                    "inference_index": calls,
                    "observation_step": source_step,
                    "observation_sim_seconds": source_step * plan["control_dt_seconds"],
                    "release_sim_seconds": source_step * plan["control_dt_seconds"],
                    "requested_latency_ms": 0.0,
                    "effective_latency_ms": 0.0,
                    "host_time_advances_simulation": False,
                    "action_sha256": digest,
                    "action_abs_max": float(np.abs(host_chunk).max()),
                }
            )
        rewards, dones, solved, finite = values
        done = False
        for offset in range(int(count)):
            if not bool(finite[offset]):
                raise RuntimeError(
                    "Kinetix produced a nonfinite state; episode is not a policy failure"
                )
            before = steps
            env.state, observation = window_states[offset], observations[offset]
            env.control_steps += 1
            steps += 1
            reward, done, success = (
                float(rewards[offset]),
                bool(dones[offset]),
                bool(solved[offset]),
            )
            total_reward += reward
            if capture:
                states.append(state_digest(env.state))
                events.append(
                    {
                        "kind": "control",
                        "control_step": before,
                        "start_sim_seconds": before * plan["control_dt_seconds"],
                        "end_sim_seconds": steps * plan["control_dt_seconds"],
                        "new_observation_step": source_step,
                        "old_observation_step": previous_source_step,
                        "new_action_index": offset,
                        "old_weights": plan["old_weights"][offset],
                        "physics_steps": env.frame_skip,
                        "reward": reward,
                        "terminal": done,
                    }
                )
            if done or success or steps == env.max_steps:
                if offset + 1 != int(count):
                    raise RuntimeError(
                        "Device window advanced beyond the first terminal boundary"
                    )
                reason = (
                    "success"
                    if success
                    else (
                        "native_terminal"
                        if done and steps < env.max_steps
                        else "budget_exhausted"
                    )
                )
                break
        if done or success or steps == env.max_steps:
            break
        previous_source_step = source_step
    result = {
        "init_state_id": seed,
        "env_seed": seed,
        "initial_observation_sha256": initial_hash,
        "success": success,
        "primitive_steps": steps,
        "max_primitive_steps": env.max_steps,
        "physics_steps": steps * env.frame_skip,
        "simulated_seconds": steps * plan["control_dt_seconds"],
        "inference_calls": calls,
        "episode_return": total_reward,
        "termination_reason": reason,
        "flow_steps": flow_steps,
        "requested_latency_ms": 0.0,
        "effective_latency_ms": 0.0,
        "delay_mapping": "native-blend",
        "step_unit": "kinetix_control_steps",
    }
    wall = time.perf_counter() - start
    return {
        "result": result,
        "wall_seconds": wall,
        "components": spans,
        "other_seconds": wall - sum(spans.values()),
        "state_hashes": states,
        "action_hashes": actions,
        "events": events,
        "full_action_host_reads": host_action_reads,
    }
