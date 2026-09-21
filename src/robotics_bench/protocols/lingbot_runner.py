"""Synchronous LingBot rollout with observed-frame KV/VAE cache updates."""

from copy import deepcopy
import time


def run_episode(
    engine, simulator, initial_observation, *, max_steps, on_frame=None, on_event=None
):
    """Preserve the native 2-frame/16-action protocol for RoboTwin.

    The first predicted frame is conditioning only. Every four executed actions
    supplies one real observation to the streaming VAE. Complete chunks update
    the model cache; terminal or budget-truncated chunks never update it with
    unexecuted actions. This is not the paper_async stale-observation protocol.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    description = simulator.description
    if not isinstance(description, str) or not description.strip():
        raise ValueError("The prepared simulator must provide a language instruction")
    engine.reset(description)
    steps = calls = cache_updates = 0
    success = False
    if on_frame:
        on_frame(simulator.render())
    while steps < max_steps:
        started = time.perf_counter()
        chunk = engine.infer_chunk(deepcopy(initial_observation), description)
        if getattr(chunk, "shape", None) != (16, 2, 16):
            raise ValueError(
                "LingBot RoboTwin requires an action array of shape (16, 2, 16)"
            )
        calls += 1
        if on_event:
            on_event(
                {
                    "kind": "inference",
                    "control_step": steps,
                    "inference_index": calls,
                    "rpc_wall_seconds": time.perf_counter() - started,
                }
            )
        keyframes = []
        for frame in range(1 if calls == 1 else 0, 2):
            for offset in range(16):
                observation, success = simulator.step(chunk[:, frame, offset].copy())
                steps += 1
                if on_frame:
                    on_frame(simulator.render())
                if success or steps == max_steps:
                    break
                if (offset + 1) % 4 == 0:
                    keyframes.append(deepcopy(observation))
            if success or steps == max_steps:
                break
        if success or steps == max_steps:
            break
        started = time.perf_counter()
        engine.update_cache(keyframes, chunk)
        cache_updates += 1
        if on_event:
            on_event(
                {
                    "kind": "cache_update",
                    "control_step": steps,
                    "observed_frames": len(keyframes),
                    "cache_update_index": cache_updates,
                    "rpc_wall_seconds": time.perf_counter() - started,
                }
            )
    return {
        "success": bool(success),
        "primitive_steps": steps,
        "max_primitive_steps": max_steps,
        "inference_calls": calls,
        "cache_updates": cache_updates,
        "step_unit": "robotwin_take_action_commands",
        "termination_reason": "success" if success else "budget_exhausted",
    }
