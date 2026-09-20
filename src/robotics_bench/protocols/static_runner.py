"""Single-environment static rollouts and primitive-step history observations.

``paper_async`` selects old observations in a serial loop. This runner does not
simulate wall-clock latency or claim concurrent inference and environment steps.
"""

from collections import deque
from copy import deepcopy
from itertools import islice
from numbers import Integral


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def run_episode(
    engine,
    simulator,
    *,
    task: str,
    init_state_id: int,
    env_seed: int,
    sampling_seed: int,
    max_steps: int,
    n_action_steps: int,
    schedule="sync",
    overlap_actions=0,
    on_frame=None,
    on_request=None,
) -> dict:
    """Execute action prefixes and return counts for one fully observed episode.

    ``task`` is the stable ledger ID; the engine receives
    ``simulator.description`` as its language instruction. Simulator reset owns
    initial-state loading and settling, which are outside ``primitive_steps``.
    Request control/observation indices start at zero, inference indices at one.
    The same explicit sampling seed is passed to each inference request.

    Callbacks run synchronously and their exceptions propagate. The caller owns
    engine/simulator closing and error artifacts; exceptions are never converted
    to a completed failed episode here.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a nonempty string")
    if not isinstance(schedule, str) or schedule not in {"sync", "paper_async"}:
        raise ValueError("schedule must be sync or paper_async")
    init_state_id = _integer(init_state_id, "init_state_id")
    env_seed = _integer(env_seed, "env_seed")
    sampling_seed = _integer(sampling_seed, "sampling_seed")
    max_steps = _integer(max_steps, "max_steps", 1)
    n_action_steps = _integer(n_action_steps, "n_action_steps", 1)
    overlap_actions = _integer(overlap_actions, "overlap_actions")
    if overlap_actions > n_action_steps:
        raise ValueError("overlap_actions must not exceed n_action_steps")
    if schedule == "sync" and overlap_actions:
        raise ValueError("sync requires overlap_actions=0")
    description = simulator.description
    if not isinstance(description, str) or not description.strip():
        raise ValueError("simulator.description must be a nonempty string")
    for name, callback in (("on_frame", on_frame), ("on_request", on_request)):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None")

    observation = simulator.reset(init_state_id, seed=env_seed)
    engine.reset((task, init_state_id, env_seed))
    if on_frame is not None:
        on_frame(simulator.render())
    history = deque(maxlen=overlap_actions + 1)
    actions = deque()
    steps = 0
    inference_calls = 0
    success = False
    reason = "budget_exhausted"

    while steps < max_steps:
        history.append((steps, deepcopy(observation)))
        if not actions:
            source_step, snapshot = (
                history[0] if len(history) > overlap_actions else history[-1]
            )
            inference_calls += 1
            if on_request is not None:
                on_request(
                    {
                        "control_step": steps,
                        "observation_step": source_step,
                        "history_offset_steps": steps - source_step,
                        "inference_index": inference_calls,
                    }
                )
            chunk = engine.infer_chunk(deepcopy(snapshot), description, sampling_seed)
            try:
                prefix = list(islice(iter(chunk), n_action_steps))
            except TypeError as exc:
                raise ValueError("engine action chunk must be iterable") from exc
            if len(prefix) != n_action_steps:
                raise ValueError("engine action chunk is shorter than n_action_steps")
            actions.extend(deepcopy(prefix))

        observation, step_success, terminated = simulator.step(actions.popleft())
        steps += 1
        success = bool(step_success)
        if on_frame is not None:
            on_frame(simulator.render())
        if success or terminated:
            reason = "success" if success else "environment_terminated"
            break

    return {
        "task": task,
        "init_state_id": init_state_id,
        "env_seed": env_seed,
        "success": success,
        "primitive_steps": steps,
        "max_primitive_steps": max_steps,
        "inference_calls": inference_calls,
        "termination_reason": reason,
    }
