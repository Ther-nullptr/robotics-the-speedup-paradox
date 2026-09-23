"""Same-task parallel evaluation with unchanged batch-one policy sampling."""

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_eval import state_digest  # noqa: E402
from device_chunk_candidate import compile_device_window  # noqa: E402
from resident_action_runner import resident_episode, validate_action_layout  # noqa: E402
from robotics_bench.kinetix.protocol import delay_plan  # noqa: E402


def seed_groups(seeds, batch_size):
    if (
        type(batch_size) is not int
        or batch_size < 1
        or not seeds
        or len(seeds) % batch_size
        or len(seeds) != len(set(seeds))
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
    ):
        raise ValueError(
            "Use unique unsigned32 seeds and a batch size that divides their count"
        )
    return [seeds[i : i + batch_size] for i in range(0, len(seeds), batch_size)]


def episode_index(episodes, seeds):
    result = {}
    for episode in episodes:
        seed = episode["result"]["env_seed"]
        if seed in result:
            raise ValueError(f"Duplicate episode seed: {seed}")
        result[seed] = episode
    if set(result) != set(seeds):
        raise ValueError("Incomplete or unexpected episode coverage")
    return result


class BatchedEvaluator:
    def __init__(self, policy, env, flow_steps):
        import jax
        from jax.custom_batching import custom_vmap

        self.policy, self.env, self.flow_steps = policy, env, flow_steps
        self.advance = compile_device_window(env, resident_actions=True)

        # JAX 0.4.35 lacks a batching rule for the barrier primitive. Apply the
        # same identity barrier to the batched tensor, without removing it.
        @custom_vmap
        def batched_barrier(value):
            return jax.lax.optimization_barrier(value)

        @batched_barrier.def_vmap
        def barrier_rule(axis_size, in_batched, value):
            del axis_size
            return jax.lax.optimization_barrier(value), in_batched[0]

        vector_window = compile_device_window(
            env, resident_actions=True, barrier=batched_barrier
        )
        self.batch_advance = jax.jit(
            jax.vmap(vector_window, in_axes=(0, 0, 0, 0, None, 0))
        )
        self.split_observations = jax.jit(
            lambda values: tuple(values[i] for i in range(values.shape[0]))
        )

    def serial(self, seeds, *, capture=False):
        start = time.perf_counter()
        rows = [
            resident_episode(
                self.policy,
                self.env,
                self.advance,
                seed=seed,
                flow_steps=self.flow_steps,
                capture=capture,
            )
            for seed in seeds
        ]
        episode_index(rows, seeds)
        return {"wall_seconds": time.perf_counter() - start, "episodes": rows}

    def vectorized(self, seeds, batch_size, *, capture=False):
        start = time.perf_counter()
        groups = seed_groups(seeds, batch_size)
        rows = []
        for group in groups:
            rows.extend(self._batch(group, capture=capture))
        episode_index(rows, seeds)
        return {"wall_seconds": time.perf_counter() - start, "episodes": rows}

    def _batch(self, seeds, *, capture):
        import jax
        import jax.numpy as jnp
        import numpy as np

        env, policy, size = self.env, self.policy, len(seeds)
        plan = delay_plan(
            0,
            physics_dt=env.physics_dt,
            frame_skip=env.frame_skip,
            execute_horizon=4,
            action_horizon=policy.action_horizon,
        )
        states, observations, noises, policy_keys, initials = [], [], [], [], []
        for seed in seeds:
            obs = env.reset(seed)
            observations.append(obs)
            states.append(env.state)
            noises.append(env._noise_rng)
            policy_keys.append(jax.random.fold_in(jax.random.key(seed), 2))
            initials.append(hashlib.sha256(np.asarray(obs).tobytes()).hexdigest())
        state = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
        noise_keys = jnp.stack(noises)
        steps, calls = [0] * size, [0] * size
        returns, success = [0.0] * size, [False] * size
        reasons, previous = ["budget_exhausted"] * size, [None] * size
        active = np.ones(size, dtype=bool)
        traces = [
            {
                "state_hashes": [],
                "action_hashes": [],
                "events": [],
                "post_terminal_windows_verified": 0,
            }
            for _ in seeds
        ]
        noop = jnp.zeros((policy.action_horizon, env.action_dim), dtype=jnp.float32)
        while active.any():
            source_steps = list(steps)
            chunks = []
            for lane in range(size):
                if active[lane]:
                    key = jax.random.fold_in(policy_keys[lane], calls[lane])
                    chunk = policy.infer_device_with_key(
                        observations[lane], self.flow_steps, key
                    )
                    validate_action_layout(
                        chunk, horizon=policy.action_horizon, action_dim=env.action_dim
                    )
                    chunks.append(chunk)
                else:
                    chunks.append(noop)
            actions = jnp.stack(chunks)
            windows, obs_windows, values, counts, valid = self.batch_advance(
                state,
                noise_keys,
                jnp.asarray(steps, dtype=jnp.int32),
                actions,
                env.params,
                jnp.asarray(active),
            )
            values, counts, valid = jax.device_get((values, counts, valid))
            if not np.all(valid[active]):
                raise ValueError(
                    "A policy produced nonfinite actions; no policy-failure result is recorded"
                )
            if np.any(counts[~active] != 0):
                raise RuntimeError("Completed environments advanced")
            if np.any(counts[active] < 1) or np.any(counts[active] > 4):
                raise RuntimeError("Invalid active control count")
            if capture:
                host_states, host_actions = jax.device_get((windows, actions))
                if (~active).any():
                    before = jax.device_get(state)
                    for lane in np.flatnonzero(~active):
                        digest = state_digest(jax.tree.map(lambda x: x[lane], before))
                        if any(
                            state_digest(jax.tree.map(lambda x: x[lane], block))
                            != digest
                            for block in host_states
                        ):
                            raise RuntimeError(
                                "A completed environment changed its state"
                            )
                        traces[lane]["post_terminal_windows_verified"] += 1
            rewards, dones, solved, finite = values
            for lane in range(size):
                if not active[lane]:
                    continue
                calls[lane] += 1
                trace = traces[lane]
                source = source_steps[lane]
                if capture:
                    action = np.asarray(host_actions[lane])
                    digest = hashlib.sha256(action.tobytes()).hexdigest()
                    trace["action_hashes"].append(digest)
                    trace["events"].append(
                        {
                            "kind": "inference",
                            "inference_index": calls[lane],
                            "observation_step": source,
                            "observation_sim_seconds": source
                            * plan["control_dt_seconds"],
                            "release_sim_seconds": source * plan["control_dt_seconds"],
                            "requested_latency_ms": 0.0,
                            "effective_latency_ms": 0.0,
                            "host_time_advances_simulation": False,
                            "action_sha256": digest,
                            "action_abs_max": float(np.abs(action).max()),
                        }
                    )
                for offset in range(int(counts[lane])):
                    if not bool(finite[lane, offset]):
                        raise RuntimeError(
                            "Nonfinite simulator state is not a policy failure"
                        )
                    before = steps[lane]
                    steps[lane] += 1
                    reward, done = (
                        float(rewards[lane, offset]),
                        bool(dones[lane, offset]),
                    )
                    success[lane] = bool(solved[lane, offset])
                    returns[lane] += reward
                    if capture:
                        trace["state_hashes"].append(
                            state_digest(
                                jax.tree.map(lambda x: x[lane], host_states[offset])
                            )
                        )
                        trace["events"].append(
                            {
                                "kind": "control",
                                "control_step": before,
                                "start_sim_seconds": before
                                * plan["control_dt_seconds"],
                                "end_sim_seconds": steps[lane]
                                * plan["control_dt_seconds"],
                                "new_observation_step": source,
                                "old_observation_step": previous[lane],
                                "new_action_index": offset,
                                "old_weights": plan["old_weights"][offset],
                                "physics_steps": env.frame_skip,
                                "reward": reward,
                                "terminal": done,
                            }
                        )
                    if done or success[lane] or steps[lane] == env.max_steps:
                        if offset + 1 != int(counts[lane]):
                            raise RuntimeError(
                                "A window advanced past its first terminal boundary"
                            )
                        active[lane] = False
                        reasons[lane] = (
                            "success"
                            if success[lane]
                            else (
                                "native_terminal"
                                if done and steps[lane] < env.max_steps
                                else "budget_exhausted"
                            )
                        )
                        break
                previous[lane] = source
            state = windows[-1]
            observations = self.split_observations(obs_windows[-1])
        # The final observation split is enqueued after the metadata transfer.
        # Close the workload timer on completed device work, including that split.
        jax.block_until_ready((state, observations))
        for lane, seed in enumerate(seeds):
            traces[lane]["result"] = {
                "init_state_id": seed,
                "env_seed": seed,
                "initial_observation_sha256": initials[lane],
                "success": success[lane],
                "primitive_steps": steps[lane],
                "max_primitive_steps": env.max_steps,
                "physics_steps": steps[lane] * env.frame_skip,
                "simulated_seconds": steps[lane] * plan["control_dt_seconds"],
                "inference_calls": calls[lane],
                "episode_return": returns[lane],
                "termination_reason": reasons[lane],
                "flow_steps": self.flow_steps,
                "requested_latency_ms": 0.0,
                "effective_latency_ms": 0.0,
                "delay_mapping": "native-blend",
                "step_unit": "kinetix_control_steps",
            }
        return traces

    def threaded(self, seeds, workers, *, capture=False, serialize_windows=False):
        """Concurrent unchanged scalar executables, sharing immutable parameters."""
        if self.flow_steps not in self.policy._compiled:
            raise ValueError("Warm the scalar policy before starting parallel workers")
        local = threading.local()
        advance = self.advance
        if serialize_windows:
            import jax

            lock = threading.Lock()

            def advance(*args):
                # Protect execution through completion, not only async enqueue.
                with lock:
                    return jax.block_until_ready(self.advance(*args))

        def one(seed):
            if not hasattr(local, "env"):
                local.env, local.policy = copy.copy(self.env), copy.copy(self.policy)
            return resident_episode(
                local.policy,
                local.env,
                advance,
                seed=seed,
                flow_steps=self.flow_steps,
                capture=capture,
            )

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, seeds))
        episode_index(rows, seeds)
        return {"wall_seconds": time.perf_counter() - start, "episodes": rows}
