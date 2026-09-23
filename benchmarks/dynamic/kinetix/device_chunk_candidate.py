"""Experimental zero-delay device chunks; native physics and each-step checks remain."""

from contextlib import contextmanager


def compile_device_window(env, *, resident_actions=False, barrier=None):
    """Compile the same native controls, optionally guarding the full GPU chunk."""
    import jax
    import jax.numpy as jnp

    barrier = jax.lax.optimization_barrier if barrier is None else barrier
    static_horizon = 4

    def advance(state, noise_rng, control_step, actions, params, enabled=True):
        def one(carry, action):
            current, step, finished = carry

            def active(_):
                key = jax.random.fold_in(noise_rng, step)
                normal = barrier(jax.random.normal(key, (env.action_dim,)))
                noise = barrier(normal * env.noise_std)
                noisy = barrier(action + noise)
                obs, new, reward, done, info = env.native.step_env(
                    key, current, noisy, params
                )
                finite = env._finite(new)
                solved = info["GoalR"]
                return (new, step + 1, done | solved | ~finite), (
                    new,
                    obs,
                    reward,
                    done,
                    solved,
                    finite,
                )

            def inactive(_):
                return carry, (
                    current,
                    env.native.get_obs(current),
                    jnp.float32(0),
                    jnp.bool_(True),
                    jnp.bool_(False),
                    jnp.bool_(True),
                )

            return jax.lax.cond(
                ~finished & (step < env.max_steps), active, inactive, operand=None
            )

        valid_actions = (
            jnp.isfinite(actions).all() if resident_actions else jnp.bool_(True)
        )
        final, outputs = jax.lax.scan(
            one,
            (
                state,
                control_step,
                (~valid_actions if resident_actions else jnp.bool_(False))
                | ~jnp.asarray(enabled),
            ),
            actions[:static_horizon] if resident_actions else actions,
        )
        states, observations, rewards, dones, solved, finite = outputs
        # Split on device to avoid a Python/JAX gather for every state leaf at
        # each host control boundary. Both timed and trace runs return these.
        states = tuple(
            jax.tree.map(lambda x: x[i], states) for i in range(static_horizon)
        )
        observations = tuple(observations[i] for i in range(static_horizon))
        result = (
            states,
            observations,
            (rewards, dones, solved, finite),
            final[1] - control_step,
        )
        return (*result, valid_actions) if resident_actions else result

    return jax.jit(advance)


def device_chunks(policy, env):
    """Prime four actions per inference, then expose only executed control states.

    This experiment only supports the native budget and zero injected latency.
    Compiling a larger region can change floating-point results; callers must
    independently compare full trajectories before claiming a valid speedup.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    original_reset, original_step, original_infer = env.reset, env.step, policy.infer
    pending = {}
    static_horizon = 4
    compiled = compile_device_window(env)

    def reset(seed):
        pending.clear()
        return original_reset(seed)

    def infer(observation, flow_steps):
        result = original_infer(observation, flow_steps)
        if pending.get("offset", 0) < pending.get("executed", 0):
            raise ValueError("Device chunk was replanned before consuming its controls")
        pending.clear()
        pending.update(actions=np.array(result[:static_horizon], copy=True), offset=0)
        return result

    def step(old, new, weights):
        del old
        if not np.all(np.asarray(weights) == 0):
            raise ValueError("Device chunk experiment requires zero injected delay")
        if "actions" not in pending:
            raise ValueError("Device chunk requires an inference before stepping")
        offset = pending["offset"]
        if offset >= static_horizon or not np.array_equal(
            new, pending["actions"][offset]
        ):
            raise ValueError("Actions do not follow the fixed execute-four schedule")
        if "states" not in pending:
            states, observations, values, count = compiled(
                env.state,
                env._noise_rng,
                jnp.int32(env.control_steps),
                jnp.asarray(pending["actions"]),
                env.params,
            )
            values, count = jax.device_get((values, count))
            pending.update(
                states=states,
                observations=observations,
                values=values,
                executed=int(count),
            )
        if offset >= pending["executed"]:
            raise ValueError("Attempted to step past a native terminal boundary")
        rewards, dones, solved, finite = pending["values"]
        if not bool(finite[offset]):
            raise RuntimeError(
                "Kinetix produced a nonfinite state; episode is not a policy failure"
            )
        env.state = pending["states"][offset]
        env.control_steps += 1
        pending["offset"] += 1
        return (
            pending["observations"][offset],
            float(rewards[offset]),
            bool(dones[offset]),
            bool(solved[offset]),
        )

    @contextmanager
    def activate():
        pending.clear()
        env.reset, env.step, policy.infer = reset, step, infer
        try:
            yield
        finally:
            env.reset, env.step, policy.infer = (
                original_reset,
                original_step,
                original_infer,
            )

    return activate
