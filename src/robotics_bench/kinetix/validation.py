"""Opt-in real-backend checks against unmodified native Kinetix stepping."""


def validate_native(env, *, seed=0, controls=4):
    import jax
    import jax.numpy as jnp
    import numpy as np

    cases = []
    for mode in ("zero_delay", "full_old_control", "constant_command"):
        env.reset(seed)
        reference = env.state
        checked = 0
        for index in range(controls):
            new = jnp.linspace(-0.4, 0.8, env.action_dim) * (1 if index % 2 else -1)
            old = new if mode == "constant_command" else -new
            weights = (
                jnp.ones(env.frame_skip)
                if mode == "full_old_control"
                else (
                    jnp.linspace(0.2, 0.8, env.frame_skip)
                    if mode == "constant_command"
                    else jnp.zeros(env.frame_skip)
                )
            )
            noise = (
                jax.random.normal(
                    jax.random.fold_in(env._noise_rng, index), (env.action_dim,)
                )
                * env.noise_std
            )
            action = old if mode == "full_old_control" else new
            want_obs, reference, want_reward, want_done, want_info = (
                env.native.step_env(
                    jax.random.key(index), reference, action + noise, env.params
                )
            )
            obs, reward, done, solved = env.step(old, new, weights)
            want, found = jax.device_get((reference, env.state))
            same_state = all(
                np.array_equal(a, b)
                for a, b in zip(
                    jax.tree.leaves(want), jax.tree.leaves(found), strict=True
                )
            )
            if not (
                same_state
                and np.array_equal(np.asarray(obs), np.asarray(want_obs))
                and reward == float(want_reward)
                and done == bool(want_done)
                and solved == bool(want_info["GoalR"])
            ):
                raise RuntimeError(
                    f"Native physics equivalence failed: {mode}, control {index}"
                )
            checked += 1
            if done:
                break
        cases.append(
            {
                "mode": mode,
                "exact_state_observation_reward_terminal_match": True,
                "control_steps_checked": checked,
                "native_physics_steps": checked * env.frame_skip,
            }
        )
    env.reset(seed)
    return {
        "status": "passed",
        "seed": seed,
        "cases": cases,
        "physics_dt_seconds": env.physics_dt,
        "frame_skip": env.frame_skip,
        "note": "Fractional blending is a command approximation; these checks establish unchanged native endpoint and constant-command behavior.",
    }
