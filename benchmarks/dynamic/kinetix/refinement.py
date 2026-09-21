"""Experimental physical substepping; no reference-state projection or replay."""

import math


def relaxation_at_substep(coefficient, factor):
    """Preserve an isolated scalar contraction over one native step."""
    if factor < 1 or int(factor) != factor or not 0 <= coefficient <= 1:
        raise ValueError("Expected an integer factor >= 1 and coefficient in [0, 1]")
    if coefficient in (0, 1) or factor == 1:
        return coefficient
    return -math.expm1(math.log1p(-coefficient) / factor)


def refined_params(base, factor, variant):
    if variant not in {
        "direct",
        "collision",
        "relaxation",
        "relaxation_drift",
        "held_motor",
        "held_motor_collision",
        "impulse_motor_collision",
    }:
        raise ValueError(f"Unknown refinement variant: {variant}")
    changes = {"dt": float(base.dt) / factor}
    if variant in {"collision", "held_motor_collision", "impulse_motor_collision"}:
        changes["baumgarte_coefficient_collision"] = (
            base.baumgarte_coefficient_collision / factor
        )
    if variant.startswith("relaxation"):
        for key in (
            "baumgarte_coefficient_collision",
            "baumgarte_coefficient_joints_p",
            "joint_stiffness",
        ):
            changes[key] = relaxation_at_substep(float(getattr(base, key)), factor)
    return base.replace(**changes)


def make_refined_env(static, base, factor, variant):
    import jax
    import jax.numpy as jnp

    from robotics_bench.kinetix.native.jax2d.engine import PhysicsEngine, select_shape
    from robotics_bench.kinetix.native.jax2d.joint import apply_motor
    from robotics_bench.kinetix.native.kinetix.environment import env as native

    params = refined_params(base, factor, variant)
    refined_static = static.replace(frame_skip=int(static.frame_skip) * factor)
    environment = native.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1", static_env_params=refined_static
    )
    if variant == "relaxation_drift" and factor > 1:

        class DriftCompensatedEngine(PhysicsEngine):
            def step(self, state, params, actions):
                state, manifolds = super().step(state, params, actions)
                # Match the native semi-implicit Euler displacement for isolated
                # constant gravity. This does not invert nonlinear contact or
                # motor dynamics and does not guarantee task equivalence.
                correction = state.gravity * params.dt * (base.dt - params.dt) / 2
                bodies = {}
                for name in ("polygon", "circle"):
                    body = getattr(state, name)
                    movable = body.active & (body.inverse_mass > 0)
                    bodies[name] = body.replace(
                        position=jnp.clip(
                            body.position + correction * movable[:, None],
                            -params.clip_position,
                            params.clip_position,
                        )
                    )
                return state.replace(**bodies), manifolds

        environment.physics_engine = DriftCompensatedEngine(refined_static)

    def advance(state, commands, params, record):
        def native_window(state, unused):
            impulses = None
            if "motor" in variant and factor > 1:
                # Keep the 60 Hz discrete servo while integrating physics at h.
                # Hold torque or retain the native start-of-window impulse.
                # Always sample from this candidate's own current state.
                def motor(index):
                    joint = jax.tree.map(lambda x: x[index], state.joint)
                    a = select_shape(state, joint.a_index, refined_static)
                    b = select_shape(state, joint.b_index, refined_static)
                    impulse = jnp.asarray(
                        apply_motor(a, b, joint, commands[index], base)
                    )
                    return impulse / factor if variant.startswith("held") else impulse

                impulses = jax.vmap(motor)(jnp.arange(static.num_joints))

            def microstep(current, micro_index):
                if impulses is None:
                    current, manifolds = environment.physics_engine.step(
                        current, params, commands
                    )
                else:
                    applied = impulses
                    if variant.startswith("impulse"):
                        applied = jnp.where(micro_index == 0, impulses, 0)
                    current, manifolds = environment.physics_engine.step(
                        current, params, commands, motor_impulses=applied
                    )
                reward, info = environment.compute_reward_info(current, manifolds)
                if record:
                    contacts = jnp.concatenate([x.active for x in manifolds])
                    return current, (reward, info["GoalR"], current, contacts)
                return current, (reward, info["GoalR"])

            state, trace = jax.lax.scan(microstep, state, xs=jnp.arange(factor))
            return state, trace

        state, trace = jax.lax.scan(
            native_window, state, xs=None, length=int(static.frame_skip)
        )
        rewards, goals = trace[:2]
        # Native reward uses the manifold at the START of each H window.
        # Keep these same two sampling times, not all 2*r microstep events.
        rewards, goals = rewards[:, 0], goals[:, 0]
        state = state.replace(timestep=state.timestep + 1)
        done = jnp.any(rewards != 0) | (state.timestep >= params.max_timesteps)
        result = (
            environment.get_obs(state),
            state,
            rewards.max(),
            done,
            {"GoalR": goals[-1]},
        )
        return (result, trace) if record else result

    # Native r=1 retains the original compilation boundary. Both recording and
    # production-style stepping execute the same physical equations.
    if factor > 1:
        environment.engine_step = jax.jit(
            lambda state, commands, params: advance(state, commands, params, False)
        )
    environment.record_physics_step = jax.jit(
        lambda state, commands, params: advance(state, commands, params, True)
    )
    return environment, params, refined_static
