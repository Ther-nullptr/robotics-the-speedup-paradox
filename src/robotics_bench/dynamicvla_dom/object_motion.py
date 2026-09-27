"""DOM-CR target-speed scaling on the native initial-velocity field.

Matches dynamic-experiments' scale_env_object_speed protocol: change only
scene.object.init_state.lin_vel, preserving direction and all other task data.
This does not prescribe a constant velocity during subsequent physics steps.
"""

from copy import deepcopy
import math


def scale_object_speed(config, scale=1.0):
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
        raise ValueError("Object speed scale must be finite and nonnegative")
    try:
        source = config["scene"]["object"]["init_state"]["lin_vel"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "DOM task must declare scene.object.init_state.lin_vel"
        ) from error
    if not isinstance(source, (list, tuple)) or len(source) != 3:
        raise ValueError("Object lin_vel must contain three finite numbers")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in source
    ):
        raise ValueError("Object lin_vel must contain three finite numbers")
    configured = [value * scale for value in source]
    source_norm, configured_norm = math.hypot(*source), math.hypot(*configured)
    if not math.isfinite(source_norm) or not math.isfinite(configured_norm):
        raise ValueError("Object speed scale or lin_vel overflows finite velocity")
    result = deepcopy(config)
    if scale != 1.0:
        result["scene"]["object"]["init_state"]["lin_vel"] = configured
    return result, dict(
        target="object",
        scope="initial_linear_velocity",
        scale=float(scale),
        source_lin_vel_mps=list(source),
        configured_lin_vel_mps=configured,
        source_speed_mps=source_norm,
        configured_speed_mps=configured_norm,
    )
