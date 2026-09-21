"""LingBot static rollouts with causal observed-frame KV/VAE cache updates."""

from collections import deque
from copy import deepcopy
import time


def paper_async_contract(schedule="sync", overlap_actions=0):
    """Declare the case-specific extension of stale inputs to stateful caches."""
    if not isinstance(schedule, str) or schedule not in {"sync", "paper_async"}:
        raise ValueError("schedule must be sync or paper_async")
    if type(overlap_actions) is not int or not 0 <= overlap_actions <= 16:
        raise ValueError("overlap_actions must be an integer in [0, 16]")
    if schedule == "sync" and overlap_actions:
        raise ValueError("sync requires overlap_actions=0")
    return {
        "protocol": "lingbot_robotwin_history_v1",
        "schedule": schedule,
        "implementation": "history_observation_with_delayed_cache",
        "overlap_actions": overlap_actions,
        "first_chunk_action_steps": 16,
        "subsequent_chunk_action_steps": 32,
        "step_unit": "robotwin_take_action_commands",
        "state_policy": "same_snapshot",
        "inference_warmup": "current_until_available",
        "cache_keyframe_interval": 4,
        "cache_observation_step": "max(0, nominal_keyframe_step - overlap_actions)",
        "cache_warmup": "pad_initial_observation",
        "cache_action_conditioning": "unchanged_generated_chunk",
        "host_execution": "serial",
        "paper_model": None,
        "paper_model_unavailable_reason": (
            "RoboTwin commands have variable duration; no fixed Tact or calibrated "
            "Tinf is supplied. RPC wall time is not a paper-model speedup."
        ),
    }


def run_episode(
    engine,
    simulator,
    initial_observation,
    *,
    max_steps,
    schedule="sync",
    overlap_actions=0,
    on_frame=None,
    on_event=None,
):
    """Preserve the native 2-frame/16-action protocol for RoboTwin.

    The first predicted frame is conditioning only. Each four-command keyframe
    uses a complete snapshot from n_prime commands earlier (initial padding
    before step zero). This shifts the entire observed KV/VAE history, which is
    the native model's input after the first inference. Generated action-cache
    conditions retain their native slots and are not measured robot state.
    Terminal/budget-truncated chunks never update the cache. The host is serial.
    """
    paper_async_contract(schedule, overlap_actions)
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    description = simulator.description
    if not isinstance(description, str) or not description.strip():
        raise ValueError("The prepared simulator must provide a language instruction")
    engine.reset(description)
    steps = calls = cache_updates = 0
    history = deque([(0, deepcopy(initial_observation))], maxlen=overlap_actions + 1)
    cache_observation_max_step = 0
    success = False
    if on_frame:
        on_frame(simulator.render())
    while steps < max_steps:
        source_step, snapshot = history[0] if steps >= overlap_actions else history[-1]
        started = time.perf_counter()
        chunk = engine.infer_chunk(deepcopy(snapshot), description)
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
                    "observation_step": source_step,
                    "history_offset_steps": steps - source_step,
                    "cache_observation_max_step": cache_observation_max_step,
                    "model_observation_source": (
                        "initial_frame" if calls == 1 else "kv_vae_cache"
                    ),
                    "inference_index": calls,
                    "rpc_wall_seconds": time.perf_counter() - started,
                }
            )
        keyframes, nominal_steps, observation_steps = [], [], []
        for frame in range(1 if calls == 1 else 0, 2):
            for offset in range(16):
                observation, success = simulator.step(chunk[:, frame, offset].copy())
                steps += 1
                history.append((steps, deepcopy(observation)))
                if on_frame:
                    on_frame(simulator.render())
                if success or steps == max_steps:
                    break
                if (offset + 1) % 4 == 0:
                    # Before n_prime steps exist, repeat step zero rather than
                    # introducing a newer frame then rewinding the streaming VAE.
                    keyframe_step, keyframe = history[0]
                    keyframes.append(deepcopy(keyframe))
                    nominal_steps.append(steps)
                    observation_steps.append(keyframe_step)
            if success or steps == max_steps:
                break
        if success or steps == max_steps:
            break
        started = time.perf_counter()
        engine.update_cache(keyframes, chunk)
        cache_updates += 1
        cache_observation_max_step = observation_steps[-1]
        if on_event:
            on_event(
                {
                    "kind": "cache_update",
                    "control_step": steps,
                    "observed_frames": len(keyframes),
                    "nominal_keyframe_steps": nominal_steps,
                    "observation_steps": observation_steps,
                    "cache_observation_max_step": cache_observation_max_step,
                    "cache_action_conditioning": "unchanged_generated_chunk",
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
        "schedule": schedule,
        "overlap_actions": overlap_actions,
        "step_unit": "robotwin_take_action_commands",
        "termination_reason": "success" if success else "budget_exhausted",
    }
