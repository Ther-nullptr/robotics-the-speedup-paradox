"""Research-only native endpoint calibration of coupled fine-step proposals.

Mixed articulated/contact cases deliberately have no resumable transition.
The auxiliary fine model disables warm starts; production physics is unchanged.
"""

from dataclasses import dataclass
import math

FIELDS = ("position", "velocity", "rotation", "angular_velocity")
MIN_SEGMENT_SECONDS = 1e-9


def partition_tick(delay, *, native_dt, frame_skip, factor, split_switch=True):
    if not math.isfinite(native_dt) or native_dt <= 0:
        raise ValueError("native_dt must be finite and positive")
    if any(type(x) is not int or x < 1 for x in (frame_skip, factor)):
        raise ValueError("frame_skip and factor must be positive integers")
    period = native_dt * frame_skip
    if not math.isfinite(delay) or not 0 <= delay <= period:
        raise ValueError("delay must lie within one native tick")
    if 0 < min(delay, period - delay) < MIN_SEGMENT_SECONDS:
        raise ValueError("delay is below the supported 1 ns endpoint resolution")
    times = {
        window * native_dt + micro * (native_dt / factor)
        for window in range(frame_skip)
        for micro in range(factor)
    }
    times.add(period)
    if split_switch:
        times.add(delay)
    times = sorted(times)
    starts = times[:-1]
    durations = [b - a for a, b in zip(times[:-1], times[1:], strict=True)]
    if any(0 < value < MIN_SEGMENT_SECONDS for value in durations):
        raise ValueError(
            "delay or grid creates a positive substep below 1 ns; no silent rounding is allowed"
        )
    count = len(durations)
    old = [start < delay for start in starts]
    old_fractions = [
        min(1.0, max(0.0, (delay - start) / duration))
        for start, duration in zip(starts, durations, strict=True)
    ]
    reward_indices = [starts.index(k * native_dt) for k in range(frame_skip)]
    while len(durations) < frame_skip * factor + 1:
        durations.append(0.0)
        starts.append(period)
        old.append(False)
        old_fractions.append(0.0)
    return {
        "starts": starts,
        "durations": durations,
        "old": old,
        "old_fractions": old_fractions,
        "switch_handling": "exact_split"
        if split_switch
        else "within_step_impulse_average",
        "reward_indices": reward_indices,
        "physical_steps": count,
        "old_fraction": delay / period,
        "control_dt_seconds": period,
        "minimum_positive_dt": min(dt for dt in durations if dt > 0),
        "minimum_supported_dt": MIN_SEGMENT_SECONDS,
    }


def calibrate_field(native_old, native_new, fine_old, fine_new, switched, fraction):
    """Continuous coordinates only; never apply this to a full state tree."""
    import numpy as np

    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must lie within [0, 1]")
    old, new, gold, gnew, actual = [
        np.asarray(x, dtype=np.float64)
        for x in (native_old, native_new, fine_old, fine_new, switched)
    ]
    one_sided = new + (actual - gnew)
    native_mix = new + fraction * (old - new)
    fine_mix = gnew + fraction * (gold - gnew)
    return one_sided, native_mix + (actual - fine_mix)


def motion_bounds(
    position, velocity, omega, acceleration, angular_acceleration, period
):
    """Conservative whole-tick bounds for a structurally isolated free body.

    The caller bounds every force/torque, not just the force at the endpoint.
    Positive Euler substeps sum to period; the envelope also bounds all prefixes.
    A margin covers ordinary float32 roundoff. This is not a contact certificate.
    """
    import numpy as np

    v = float(np.max(np.abs(velocity))) + abs(acceleration) * period
    p = float(np.max(np.abs(position))) + period * v
    w = abs(float(omega)) + abs(angular_acceleration) * period
    return {
        name: value * 1.00001 + 1e-6
        for name, value in (("position", p), ("velocity", v), ("angular_velocity", w))
    }


def continuous_state(state):
    import jax
    import numpy as np

    state = jax.device_get(state)
    return {
        name: {key: np.asarray(getattr(getattr(state, name), key)) for key in FIELDS}
        for name in ("polygon", "circle")
    }


def state_equal(left, right):
    import jax
    import numpy as np

    return jax.tree.structure(left) == jax.tree.structure(right) and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


@dataclass
class CalibrationProbe:
    proposal: dict
    one_sided_proposal: dict | None
    diagnostics: dict
    transition: object = None

    def require_transition(self):
        if self.transition is None:
            raise RuntimeError(
                "Proposal is not a validated transition: "
                + ", ".join(self.diagnostics["unsupported_reasons"])
            )
        return self.transition


class ResidualCalibrator:
    def __init__(self, static, params, *, factor=2, fine_model="split"):
        import jax
        import jax.numpy as jnp
        from .native.kinetix.environment import env as native
        from .native.jax2d.engine import select_shape
        from .native.jax2d.joint import apply_motor

        if fine_model not in {"split", "impulse-blend"}:
            raise ValueError("Unknown auxiliary fine model")
        self.fine_model = fine_model

        partition_tick(
            0,
            native_dt=float(params.dt),
            frame_skip=int(static.frame_skip),
            factor=factor,
        )
        self.static, self.params, self.factor = static, params, factor
        self.period = float(params.dt) * int(static.frame_skip)
        self.native = native.make_kinetix_env_from_name(
            "Kinetix-Symbolic-Continuous-v1", static_env_params=static
        )
        fine_static = static.replace(do_warm_starting=False)
        self.fine = native.make_kinetix_env_from_name(
            "Kinetix-Symbolic-Continuous-v1", static_env_params=fine_static
        )
        pair_count = sum(
            len(x)
            for x in (
                self.fine.physics_engine.poly_poly_pairs,
                self.fine.physics_engine.circle_poly_pairs,
                self.fine.physics_engine.circle_circle_pairs,
            )
        )

        def fine_step(
            state, commands, durations, old_command, new_command, old_weights
        ):
            state = self.clear_caches(state)

            def advance(current, item):
                command, duration, weight = item

                def positive(current):
                    step_params = params.replace(
                        dt=duration,
                        baumgarte_coefficient_collision=params.baumgarte_coefficient_collision
                        * duration
                        / params.dt,
                    )
                    if fine_model == "impulse-blend":

                        def motor(index):
                            joint = jax.tree.map(lambda x: x[index], current.joint)
                            a = select_shape(current, joint.a_index, static)
                            b = select_shape(current, joint.b_index, static)
                            old_impulse = jnp.asarray(
                                apply_motor(
                                    a, b, joint, old_command[index], step_params
                                )
                            )
                            new_impulse = jnp.asarray(
                                apply_motor(
                                    a, b, joint, new_command[index], step_params
                                )
                            )
                            mixed = old_impulse + (new_impulse - old_impulse) * (
                                1 - weight
                            )
                            return jnp.where(
                                weight == 0,
                                new_impulse,
                                jnp.where(weight == 1, old_impulse, mixed),
                            )

                        impulses = jax.vmap(motor)(jnp.arange(static.num_joints))
                        current, manifolds = self.fine.physics_engine.step(
                            current, step_params, command, motor_impulses=impulses
                        )
                    else:
                        current, manifolds = self.fine.physics_engine.step(
                            current, step_params, command
                        )
                    reward, info = self.fine.compute_reward_info(current, manifolds)
                    bits = jnp.concatenate([x.active for x in manifolds])
                    return current, (reward, info["GoalR"], bits)

                def zero(current):
                    return current, (
                        jnp.asarray(0.0),
                        jnp.asarray(False),
                        jnp.zeros(pair_count, dtype=bool),
                    )

                return jax.lax.cond(duration > 0, positive, zero, current)

            state, trace = jax.lax.scan(
                advance, state, (commands, durations, old_weights)
            )
            return state.replace(timestep=state.timestep + 1), trace

        self._fine_step = jax.jit(fine_step)

        def geometry(state):
            manifolds = self.native.physics_engine.calculate_collision_manifolds(
                self.clear_caches(state)
            )
            return jnp.concatenate(
                (
                    manifolds[0].active.any(axis=-1),
                    manifolds[1].active,
                    manifolds[2].active,
                )
            )

        self._geometry = jax.jit(geometry)
        self._finite = jax.jit(
            lambda state: jax.tree.reduce(
                jnp.logical_and,
                jax.tree.map(lambda x: jnp.isfinite(x).all(), state),
                True,
            )
        )

    @staticmethod
    def clear_caches(state):
        import jax
        import jax.numpy as jnp

        return state.replace(
            joint=state.joint.replace(
                acc_impulse=jnp.zeros_like(state.joint.acc_impulse),
                acc_r_impulse=jnp.zeros_like(state.joint.acc_r_impulse),
            ),
            **{
                name: jax.tree.map(jnp.zeros_like, getattr(state, name))
                for name in ("acc_rr_manifolds", "acc_cr_manifolds", "acc_cc_manifolds")
            },
        )

    def _with_body_fields(self, carrier, values):
        import jax.numpy as jnp

        return carrier.replace(
            **{
                name: getattr(carrier, name).replace(
                    **{
                        key: jnp.asarray(
                            values[name][key],
                            dtype=getattr(getattr(carrier, name), key).dtype,
                        )
                        for key in FIELDS
                    }
                )
                for name in values
            }
        )

    def _refresh_thrusters(self, state):
        import jax
        from .native.jax2d.engine import select_shape
        from .native.jax2d.maths import rmat

        def refresh(thruster):
            body = select_shape(state, thruster.object_index, self.static)
            return thruster.replace(
                global_position=body.position
                + rmat(body.rotation) @ thruster.relative_position
            )

        return state.replace(thruster=jax.vmap(refresh)(state.thruster))

    def probe(self, state, old, new, delay, *, audit=True):
        import jax
        import jax.numpy as jnp
        import numpy as np

        plan = partition_tick(
            delay,
            native_dt=float(self.params.dt),
            frame_skip=int(self.static.frame_skip),
            factor=self.factor,
            split_switch=self.fine_model == "split",
        )
        expected = self.native.action_space(self.params).shape
        if any(
            np.shape(value) != expected or not np.isfinite(np.asarray(value)).all()
            for value in (old, new)
        ):
            raise ValueError(
                "Actions must be finite arrays with the native action shape"
            )
        old, new = jnp.asarray(old), jnp.asarray(new)
        old_command = self.native.action_type.process_action(old, state, self.static)
        new_command = self.native.action_type.process_action(new, state, self.static)
        same = np.array_equal(np.asarray(old_command), np.asarray(new_command))
        endpoint = (
            "native_old_exact"
            if delay == self.period
            else "native_new_exact"
            if same or delay == 0
            else None
        )
        if endpoint and not audit:
            result = self.native.step_env(
                jax.random.key(0),
                state,
                old if endpoint == "native_old_exact" else new,
                self.params,
            )
            if not bool(self._finite(result[1])):
                raise RuntimeError("Native endpoint produced a nonfinite state")
            return CalibrationProbe(
                continuous_state(result[1]),
                None,
                {
                    "branch": endpoint,
                    "resumable": True,
                    "unsupported_reasons": [],
                    "plan": plan,
                    "counterfactual_audit": False,
                    "fine_model": self.fine_model,
                    "counterfactual_fine_physics_steps": 0,
                    "native_physics_steps": int(self.static.frame_skip),
                    "calibrated_task_reward_available": True,
                },
                result,
            )
        native_old = self.native.step_env(jax.random.key(0), state, old, self.params)
        native_new = self.native.step_env(jax.random.key(0), state, new, self.params)
        count = len(plan["durations"])
        old_commands = jnp.broadcast_to(old_command, (count, old_command.size))
        new_commands = jnp.broadcast_to(new_command, (count, new_command.size))
        weights = jnp.asarray(plan["old_fractions"], dtype=jnp.float32)
        switched = old_commands + (new_commands - old_commands) * (1 - weights[:, None])
        switched = jnp.where(
            weights[:, None] == 0,
            new_commands,
            jnp.where(weights[:, None] == 1, old_commands, switched),
        )
        durations = jnp.asarray(plan["durations"], dtype=jnp.float32)
        fine_old, old_trace = self._fine_step(
            state,
            old_commands,
            durations,
            old_command,
            new_command,
            jnp.ones_like(weights),
        )
        fine_new, new_trace = self._fine_step(
            state,
            new_commands,
            durations,
            old_command,
            new_command,
            jnp.zeros_like(weights),
        )
        fine_actual, actual_trace = self._fine_step(
            state, switched, durations, old_command, new_command, weights
        )
        states = (native_old[1], native_new[1], fine_old, fine_new, fine_actual)
        if not all(bool(self._finite(value)) for value in states):
            raise RuntimeError("A counterfactual produced a nonfinite state")
        fields = [continuous_state(value) for value in states]
        one, two = {}, {}
        for name in ("polygon", "circle"):
            body = getattr(state, name)
            one[name], two[name] = {}, {}
            for key in FIELDS:
                single, balanced = calibrate_field(
                    *(value[name][key] for value in fields), plan["old_fraction"]
                )
                movable = np.asarray(
                    body.active
                    & (
                        (body.inverse_mass != 0)
                        if key in ("position", "velocity")
                        else (body.inverse_inertia != 0)
                    )
                )
                if single.ndim == 2:
                    movable = movable[:, None]
                one[name][key] = np.where(movable, single, fields[1][name][key])
                two[name][key] = np.where(movable, balanced, fields[1][name][key])
        if not all(
            np.isfinite(x).all()
            for value in (one, two)
            for body in value.values()
            for x in body.values()
        ):
            raise RuntimeError("Calibration produced a nonfinite proposal")
        carrier = self._with_body_fields(fine_actual, two)
        geometries = [np.asarray(self._geometry(value)) for value in (*states, carrier)]
        active_count = sum(
            int(np.asarray(getattr(state, name).active).sum())
            for name in ("polygon", "circle")
        )
        reasons = []
        if active_count != 1:
            reasons.append("not_a_single_isolated_body")
        if (
            sum(
                int(
                    np.asarray(
                        getattr(state, name).active
                        & (getattr(state, name).inverse_mass > 0)
                    ).sum()
                )
                for name in ("polygon", "circle")
            )
            != 1
        ):
            reasons.append("not_a_single_dynamic_body")
        if bool(jnp.any(state.joint.active)):
            reasons.append("joint_cache_transfer_unvalidated")
        all_traces = [
            jax.device_get(trace) for trace in (old_trace, new_trace, actual_trace)
        ]
        if any(x.any() for x in geometries) or any(x[2].any() for x in all_traces):
            reasons.append("contact_or_event_reconstruction_unvalidated")
        cache_values = [state.joint.acc_impulse, state.joint.acc_r_impulse]
        for name in ("acc_rr_manifolds", "acc_cr_manifolds", "acc_cc_manifolds"):
            cache_values.extend(
                (
                    getattr(state, name).acc_impulse_normal,
                    getattr(state, name).acc_impulse_tangent,
                )
            )
        if any(np.any(np.asarray(x) != 0) for x in cache_values):
            reasons.append("nonzero_input_impulse_cache")
        envelope = None
        if active_count == 1 and not bool(jnp.any(state.joint.active)):
            active = np.concatenate(
                [
                    np.asarray(getattr(state, name).active)
                    for name in ("polygon", "circle")
                ]
            )
            body_index = int(np.flatnonzero(active)[0])
            name = "polygon" if body_index < self.static.num_polygons else "circle"
            local_index = (
                body_index
                if name == "polygon"
                else body_index - self.static.num_polygons
            )
            body = getattr(state, name)
            live_thrusters = np.asarray(state.thruster.active)
            if (
                self.factor != 2
                or int(self.static.frame_skip) != 2
                or int(live_thrusters.sum()) > 1
            ):
                reasons.append("outside_tested_isolated_continuation_domain")
            parents = np.asarray(state.thruster.object_index)[live_thrusters]
            if np.any(parents != body_index):
                reasons.append("unsupported_thruster_parent")
            forces = np.abs(np.asarray(state.thruster.power)[live_thrusters]) * abs(
                self.params.base_thruster_power
            )
            arms = np.linalg.norm(
                np.asarray(state.thruster.relative_position)[live_thrusters], axis=-1
            )
            acceleration = float(np.linalg.norm(np.asarray(state.gravity))) + float(
                forces.sum()
            ) * abs(float(body.inverse_mass[local_index]))
            angular_acceleration = float((forces * arms).sum()) * abs(
                float(body.inverse_inertia[local_index])
            )
            envelope = motion_bounds(
                np.asarray(body.position[local_index]),
                np.asarray(body.velocity[local_index]),
                float(body.angular_velocity[local_index]),
                acceleration,
                angular_acceleration,
                self.period,
            )
            if any(
                envelope[key] >= limit
                for key, limit in (
                    ("position", self.params.clip_position),
                    ("velocity", self.params.clip_velocity),
                    ("angular_velocity", self.params.clip_angular_velocity),
                )
            ):
                reasons.append("whole_tick_motion_envelope_may_clip")
        for key, limit in (
            ("position", self.params.clip_position),
            ("velocity", self.params.clip_velocity),
            ("angular_velocity", self.params.clip_angular_velocity),
        ):
            if any(
                np.any(
                    np.abs(value[name][key][np.asarray(getattr(state, name).active)])
                    >= limit
                )
                for value in (*fields, two, continuous_state(carrier))
                for name in ("polygon", "circle")
            ):
                reasons.append("clipping_or_bound_extrapolation")
                break
        branch = "proposal_only"
        transition = None
        if delay == self.period:
            transition, branch = native_old, "native_old_exact"
        elif same or delay == 0:
            transition, branch = native_new, "native_new_exact"
        elif not reasons:
            carrier = self._refresh_thrusters(self.clear_caches(carrier))
            if not bool(self._finite(carrier)):
                raise RuntimeError("Calibrated float32 continuation state is nonfinite")
            transition = (
                self.native.get_obs(carrier),
                carrier,
                jnp.asarray(0.0),
                carrier.timestep >= self.params.max_timesteps,
                {
                    "GoalR": jnp.asarray(False),
                    "distance": jnp.asarray(0.0),
                    "rr_manifolds": None,
                    "cr_manifolds": None,
                },
            )
            branch = "isolated_body_calibrated"
        diagnostics = {
            "branch": branch,
            "resumable": transition is not None,
            "unsupported_reasons": []
            if branch.startswith("native_")
            else sorted(set(reasons)),
            "plan": plan,
            "counterfactual_audit": True,
            "isolated_motion_bounds": envelope,
            "fine_warm_starting": False,
            "fine_model": self.fine_model,
            "counterfactual_fine_physics_steps": 3 * plan["physical_steps"],
            "native_physics_steps": 2 * int(self.static.frame_skip),
            "proposal_correction_max_abs_by_field": {
                key: max(
                    float(np.max(np.abs(two[name][key] - fields[4][name][key])))
                    for name in two
                    if two[name][key].size
                )
                for key in FIELDS
            },
            "fine_contact_trace_disagreements": int(
                np.count_nonzero(all_traces[2][2] != all_traces[1][2])
            ),
            "proposal_vs_fine_endpoint_contact_disagreements": int(
                np.count_nonzero(geometries[-1] != geometries[4])
            ),
            "fine_sampled_rewards": all_traces[2][0][plan["reward_indices"]].tolist(),
            "calibrated_task_reward_available": transition is not None,
        }
        return CalibrationProbe(two, one, diagnostics, transition)
