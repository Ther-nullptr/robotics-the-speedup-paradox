"""Owned native level loading; training and cloud checkpoint helpers are excluded."""

import json
import pickle
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp

from robotics_bench.kinetix.native.jax2d.engine import calculate_collision_matrix
from robotics_bench.kinetix.native.kinetix.environment.env_state import (
    EnvState,
    StaticEnvParams,
    EnvParams,
)


def load_pcg_state_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def import_env_state_from_json(
    json_file: dict[str, Any],
) -> tuple[EnvState, StaticEnvParams, EnvParams]:
    from robotics_bench.kinetix.native.kinetix.environment.env import create_empty_env

    def normalise(k, v):
        if k == "screen_dim":
            return v
        if type(v) is dict and "0" in v:
            return jnp.array([normalise(k, v[str(i)]) for i in range(len(v))])
        return v

    env_state = json_file["env_state"]
    env_params = json_file["env_params"]
    static_env_params = json_file["static_env_params"]
    env_params_target = EnvParams()
    static_env_params_target = StaticEnvParams()
    new_env_params = flax.serialization.from_state_dict(
        env_params_target, {k: normalise(k, v) for k, v in env_params.items()}
    )
    norm_static = {k: normalise(k, v) for k, v in static_env_params.items()}
    # norm_static["screen_dim"] = tuple(static_env_params_target.screen_dim)
    norm_static["downscale"] = static_env_params_target.downscale
    # print(
    #     static_env_params_target,
    # )
    new_static_env_params = flax.serialization.from_state_dict(
        static_env_params_target, norm_static
    )
    new_static_env_params = new_static_env_params.replace(
        screen_dim=static_env_params_target.screen_dim
    )

    env_state_target = create_empty_env(new_static_env_params)

    def astype(x, all):
        return jnp.astype(x, all.dtype)

    def _load_rigidbody(env_state_target, i, is_poly):

        to_load_from: dict[str, Any] = env_state[
            "circle" if not is_poly else "polygon"
        ][i]
        role = to_load_from.pop("role")
        density = to_load_from.pop("density")
        if "highlighted" in to_load_from:
            _ = to_load_from.pop("highlighted")
        new_obj = flax.serialization.from_state_dict(
            jax.tree.map(
                lambda x: x[i],
                env_state_target.circle if not is_poly else env_state_target.polygon,
            ),
            {k: normalise(k, v) for k, v in to_load_from.items()},
        )

        if is_poly:
            env_state_target = env_state_target.replace(
                polygon_shape_roles=env_state_target.polygon_shape_roles.at[i].set(
                    role
                ),
                polygon_densities=env_state_target.polygon_densities.at[i].set(density),
                polygon=jax.tree.map(
                    lambda all, new: all.at[i].set(astype(new, all)),
                    env_state_target.polygon,
                    new_obj,
                ),
            )
        else:
            env_state_target = env_state_target.replace(
                circle_shape_roles=env_state_target.circle_shape_roles.at[i].set(role),
                circle_densities=env_state_target.circle_densities.at[i].set(density),
                circle=jax.tree.map(
                    lambda all, new: all.at[i].set(astype(new, all)),
                    env_state_target.circle,
                    new_obj,
                ),
            )
        return env_state_target

    # Now load the env state:
    for i in range(new_static_env_params.num_circles):
        env_state_target = _load_rigidbody(env_state_target, i, False)
    for i in range(new_static_env_params.num_polygons):
        env_state_target = _load_rigidbody(env_state_target, i, True)

    for i in range(new_static_env_params.num_joints):
        to_load_from = env_state["joint"][i]
        motor_binding = to_load_from.pop("motor_binding")
        new_obj = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.joint),
            {k: normalise(k, v) for k, v in to_load_from.items()},
        )
        env_state_target = env_state_target.replace(
            joint=jax.tree.map(
                lambda all, new: all.at[i].set(astype(new, all)),
                env_state_target.joint,
                new_obj,
            ),
            motor_bindings=env_state_target.motor_bindings.at[i].set(motor_binding),
        )

    for i in range(new_static_env_params.num_thrusters):
        to_load_from = env_state["thruster"][i]
        thruster_binding = to_load_from.pop("thruster_binding")
        new_obj = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.thruster),
            {k: normalise(k, v) for k, v in to_load_from.items()},
        )

        env_state_target = env_state_target.replace(
            thruster=jax.tree.map(
                lambda all, new: all.at[i].set(astype(new, all)),
                env_state_target.thruster,
                new_obj,
            ),
            thruster_bindings=env_state_target.thruster_bindings.at[i].set(
                thruster_binding
            ),
        )

    env_state_target = env_state_target.replace(
        collision_matrix=flax.serialization.from_state_dict(
            env_state_target.collision_matrix,
            normalise("collision_matrix", env_state["collision_matrix"]),
        )
    )

    for i in range(env_state_target.acc_rr_manifolds.active.shape[0]):
        a = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.acc_rr_manifolds),
            {k: normalise(k, v) for k, v in env_state["acc_rr_manifolds"][i].items()},
        )
        b = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.acc_rr_manifolds),
            {
                k: normalise(k, v)
                for k, v in env_state["acc_rr_manifolds"][i + 1].items()
            },
        )
        env_state_target = env_state_target.replace(
            acc_rr_manifolds=jax.tree.map(
                lambda all, new: all.at[i].set(astype(new, all)),
                env_state_target.acc_rr_manifolds,
                a,
            ),
        )
        env_state_target.replace(
            acc_rr_manifolds=jax.tree.map(
                lambda all, new: all.at[i + 1].set(astype(new, all)),
                env_state_target.acc_rr_manifolds,
                b,
            )
        )
    for i in range(env_state_target.acc_cr_manifolds.active.shape[0]):
        a = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.acc_cr_manifolds),
            {k: normalise(k, v) for k, v in env_state["acc_cr_manifolds"][i].items()},
        )
        env_state_target = env_state_target.replace(
            acc_cr_manifolds=jax.tree.map(
                lambda all, new: all.at[i].set(astype(new, all)),
                env_state_target.acc_cr_manifolds,
                a,
            ),
        )
    for i in range(env_state_target.acc_cc_manifolds.active.shape[0]):
        a = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.acc_cc_manifolds),
            {k: normalise(k, v) for k, v in env_state["acc_cc_manifolds"][i].items()},
        )
        env_state_target = env_state_target.replace(
            acc_cc_manifolds=jax.tree.map(
                lambda all, new: all.at[i].set(astype(new, all)),
                env_state_target.acc_cc_manifolds,
                a,
            ),
        )

    env_state_target = env_state_target.replace(
        collision_matrix=calculate_collision_matrix(
            new_static_env_params, env_state_target.joint
        )
    )

    return (
        env_state_target,
        new_static_env_params,
        new_env_params.replace(max_timesteps=env_params_target.max_timesteps),
    )


def load_from_json_file(filename):
    with open(filename, "r") as f:
        return import_env_state_from_json(json.load(f))
