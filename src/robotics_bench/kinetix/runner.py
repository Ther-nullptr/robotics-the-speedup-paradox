"""One episode with native-time latency replay, without policy prefetch or autoreset."""

import hashlib
import time

from .protocol import delay_plan


def run_episode(
    policy,
    env,
    *,
    seed,
    flow_steps,
    latency_ms,
    execute_horizon=4,
    mapping="fine",
    max_steps=None,
    on_event=None,
    on_frame=None,
):
    import numpy as np

    for name, value in (("seed", seed), ("flow_steps", flow_steps)):
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid {name}")
    if seed >= 2**32:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    budget = env.max_steps if max_steps is None else max_steps
    if type(budget) is not int or not 0 < budget <= env.max_steps:
        raise ValueError("max_steps must be within the native episode budget")
    plan = delay_plan(
        latency_ms,
        physics_dt=env.physics_dt,
        frame_skip=env.frame_skip,
        execute_horizon=execute_horizon,
        action_horizon=policy.action_horizon,
        mapping=mapping,
    )
    weights = np.asarray(plan["old_weights"], dtype=np.float32)
    observation = env.reset(seed)
    initial_hash = hashlib.sha256(np.asarray(observation).tobytes()).hexdigest()
    policy.reset(seed)
    previous = np.zeros((policy.action_horizon, env.action_dim), dtype=np.float32)
    previous_source_step = None
    steps = calls = 0
    total_reward = 0.0
    success = False
    reason = "budget_exhausted"
    if on_frame:
        on_frame(env.render())
    while steps < budget:
        source_step = steps
        started = time.perf_counter()
        chunk = np.array(policy.infer(observation, flow_steps), copy=True)
        elapsed = time.perf_counter() - started
        if (
            chunk.shape != previous.shape
            or chunk.dtype.kind != "f"
            or not np.isfinite(chunk).all()
        ):
            raise ValueError(
                "Policy must return a finite floating-point action chunk of the declared shape"
            )
        calls += 1
        if on_event:
            on_event(
                {
                    "kind": "inference",
                    "inference_index": calls,
                    "observation_step": source_step,
                    "observation_sim_seconds": source_step * plan["control_dt_seconds"],
                    "release_sim_seconds": source_step * plan["control_dt_seconds"]
                    + plan["effective_latency_ms"] / 1000,
                    "requested_latency_ms": plan["requested_latency_ms"],
                    "effective_latency_ms": plan["effective_latency_ms"],
                    "host_policy_call_seconds": elapsed,
                    "host_time_advances_simulation": False,
                    "action_sha256": hashlib.sha256(chunk.tobytes()).hexdigest(),
                    "action_abs_max": float(np.abs(chunk).max()),
                }
            )
        done = False
        for offset in range(execute_horizon):
            before = steps
            observation, reward, done, solved = env.step(
                previous[offset], chunk[offset], weights[offset]
            )
            steps += 1
            total_reward += float(reward)
            success = bool(solved)
            if on_event:
                on_event(
                    {
                        "kind": "control",
                        "control_step": before,
                        "start_sim_seconds": before * plan["control_dt_seconds"],
                        "end_sim_seconds": steps * plan["control_dt_seconds"],
                        "new_observation_step": source_step,
                        "old_observation_step": previous_source_step,
                        "new_action_index": offset,
                        "old_weights": weights[offset].tolist(),
                        "physics_steps": env.frame_skip,
                        "reward": float(reward),
                        "terminal": bool(done),
                    }
                )
            if on_frame:
                on_frame(env.render())
            if done or success or steps == budget:
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
        if done or success or steps == budget:
            break
        previous = np.concatenate(
            (chunk[execute_horizon:], np.zeros_like(chunk[:execute_horizon]))
        )
        previous_source_step = source_step
    return {
        "init_state_id": seed,
        "env_seed": seed,
        "initial_observation_sha256": initial_hash,
        "success": success,
        "primitive_steps": steps,
        "max_primitive_steps": budget,
        "physics_steps": steps * env.frame_skip,
        "simulated_seconds": steps * plan["control_dt_seconds"],
        "inference_calls": calls,
        "episode_return": total_reward,
        "termination_reason": reason,
        "flow_steps": flow_steps,
        "requested_latency_ms": plan["requested_latency_ms"],
        "effective_latency_ms": plan["effective_latency_ms"],
        "delay_mapping": mapping,
        "step_unit": "kinetix_control_steps",
    }
