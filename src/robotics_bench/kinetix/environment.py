"""Native Kinetix physics with processed-command latency blending.

The control-step reward/termination structure follows the owned copy of
FLAIROx/Kinetix env.py (copyright 2024 Michael Matthews, MIT; see KINETIX_LICENSE).
Its Jax2D physics core is also maintained in this repository. No finer timestep
is introduced.
"""

from dataclasses import asdict
from pathlib import Path

from .protocol import mix_processed_commands


class KinetixEnvironment:
    def __init__(self, level, *, action_noise_std=0.1):
        import jax
        import jax.numpy as jnp

        from .native.kinetix.environment import env as native
        from .native.kinetix.util.saving import load_from_json_file

        self.level, static, params = load_from_json_file(
            str(Path(__file__).parent / "levels" / f"{level}.json")
        )
        self.native = native.make_kinetix_env_from_name(
            "Kinetix-Symbolic-Continuous-v1", static_env_params=static
        )
        self.params, self.static = params, static
        self.physics_dt, self.frame_skip = float(params.dt), int(static.frame_skip)
        self.max_steps = int(params.max_timesteps)
        self.action_dim = int(self.native.action_space(params).shape[0])
        self.noise_std = action_noise_std
        self._renderer = None
        self.metadata = {
            "env_params": asdict(params),
            "source_file": str(Path(native.__file__).resolve()),
            "static_env_params": asdict(static),
            "observation_type": "symbolic",
            "action_type": "continuous",
            "action_noise_std": action_noise_std,
            "action_noise_cadence": "control_step",
            "blend_domain": "processed_actuator_command",
            "terminal_cadence": "native_control_boundary",
        }

        def advance(state, commands):
            def physics_step(current, command):
                current, manifolds = self.native.physics_engine.step(
                    current, params, command
                )
                reward, info = self.native.compute_reward_info(current, manifolds)
                return current, (reward, info["GoalR"])

            state, (rewards, goals) = jax.lax.scan(physics_step, state, commands)
            state = state.replace(timestep=state.timestep + 1)
            done = jnp.any(rewards != 0) | (state.timestep >= params.max_timesteps)
            # Preserve native max-reward / last-slot GoalR semantics. The native
            # solver completes frame_skip slots before checking termination.
            return (
                self.native.get_obs(state),
                state,
                rewards.max(),
                done,
                goals[-1],
            )

        self._advance = jax.jit(advance)
        self._finite = jax.jit(
            lambda state: jax.tree.reduce(
                jnp.logical_and,
                jax.tree.map(lambda x: jnp.isfinite(x).all(), state),
                True,
            )
        )

    def reset(self, seed):
        import jax

        obs, self.state = self.native.reset_to_level(
            jax.random.fold_in(jax.random.key(seed), 0), self.level, self.params
        )
        self._noise_rng = jax.random.fold_in(jax.random.key(seed), 1)
        self.control_steps = 0
        self.observation_dim = int(obs.shape[-1])
        return obs

    def step(self, old, new, weights):
        import jax
        import jax.numpy as jnp
        import numpy as np

        if len(weights) != self.frame_skip:
            raise ValueError("Expected exactly the native frame_skip command slots")
        key = jax.random.fold_in(self._noise_rng, self.control_steps)
        noise = jax.random.normal(key, (self.action_dim,)) * self.noise_std
        weights = np.asarray(weights)
        # Preserve the native compilation boundary for complete old/new and
        # constant-command controls. Fusing action processing into the physics
        # scan can change floating-point rounding even when delay is zero.
        if np.all(weights == 0) or np.all(weights == 1) or np.array_equal(old, new):
            action = old if np.all(weights == 1) else new
            obs, state, reward, done, info = self.native.step_env(
                key, self.state, jnp.asarray(action) + noise, self.params
            )
            solved = info["GoalR"]
        else:
            old_command = self.native.action_type.process_action(
                jnp.asarray(old) + noise, self.state, self.static
            )
            new_command = self.native.action_type.process_action(
                jnp.asarray(new) + noise, self.state, self.static
            )
            commands = mix_processed_commands(old_command, new_command, weights, jnp)
            obs, state, reward, done, solved = self._advance(self.state, commands)
        finite = self._finite(state)
        reward, done, solved, finite = jax.device_get((reward, done, solved, finite))
        if not bool(finite):
            raise RuntimeError(
                "Kinetix produced a nonfinite state; episode is not a policy failure"
            )
        self.state = state
        self.control_steps += 1
        return obs, float(reward), bool(done), bool(solved)

    def render(self):
        import jax
        import numpy as np

        if self._renderer is None:
            from .native.kinetix.render.renderer_pixels import make_render_pixels

            self._renderer = jax.jit(make_render_pixels(self.params, self.static))
        return np.asarray(jax.device_get(self._renderer(self.state))).astype(np.uint8)
