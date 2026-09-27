"""Virtual action availability on an unchanged native physics grid."""

import math


DELAY_MODES = ("coarse", "fine")
DELAY_ALIASES = {"legacy-round": "coarse", "native-blend": "fine"}


def normalize_mapping(mapping):
    """Resolve historical names without changing their recorded provenance."""
    mode = DELAY_ALIASES.get(mapping, mapping)
    if mode not in DELAY_MODES:
        raise ValueError("Delay mode must be coarse or fine")
    return mode


LEVELS = (
    "grasp_easy",
    "catapult",
    "cartpole_thrust",
    "hard_lunar_lander",
    "mjc_half_cheetah",
    "mjc_swimmer",
    "mjc_walker",
    "h17_unicycle",
    "chain_lander",
    "catcher_v3",
    "trampoline",
    "car_launch",
)


def delay_plan(
    latency_ms,
    *,
    physics_dt,
    frame_skip,
    execute_horizon,
    action_horizon=8,
    mapping="fine",
):
    """Keep native solver calls fixed; only change old/new command weights.

    fine is a time-average command approximation inside a physics slot,
    not exact integration of a mid-slot state change. coarse intentionally
    retains Python's nearest-even rounding to complete native control periods.
    """
    if not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("physics_dt must be finite and positive")
    for name, value in (
        ("frame_skip", frame_skip),
        ("execute_horizon", execute_horizon),
        ("action_horizon", action_horizon),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if execute_horizon > action_horizon:
        raise ValueError("Execution horizon exceeds the predicted action horizon")
    mode = normalize_mapping(mapping)
    latency_ms = float(latency_ms)
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms must be finite and nonnegative")
    control_ms = physics_dt * frame_skip * 1000
    limit = min(execute_horizon, action_horizon - execute_horizon) * control_ms
    if latency_ms > limit + 1e-9:
        raise ValueError(
            "Delay exceeds the execution horizon or available old-action tail"
        )
    effective = (
        round(latency_ms / control_ms) * control_ms if mode == "coarse" else latency_ms
    )
    slots = execute_horizon * frame_skip
    q = effective / (physics_dt * 1000)
    flat = [min(1.0, max(0.0, q - k)) for k in range(slots)]
    return {
        "mapping": mapping,
        "requested_latency_ms": latency_ms,
        "effective_latency_ms": effective,
        "physics_dt_seconds": physics_dt,
        "control_dt_seconds": physics_dt * frame_skip,
        "physics_steps_per_cycle": slots,
        "old_weights": [flat[k : k + frame_skip] for k in range(0, slots, frame_skip)],
    }


def mix_processed_commands(old, new, weights, xp):
    """Mix actuator commands with exact endpoints and constant-command identity."""
    weights = xp.asarray(weights)[..., None]
    mixed = old + (new - old) * (1 - weights)
    return xp.where(weights == 1, old, xp.where(weights == 0, new, mixed))
