"""Load local checkpoint parameters into this repository's RTC Flax execution."""

from pathlib import Path
import pickle


class KinetixFlowPolicy:
    def __init__(self, checkpoint, *, observation_dim, action_dim):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        from . import flow_model as module

        self.config = module.ModelConfig()
        self.action_horizon = self.config.action_chunk_size
        policy = module.FlowPolicy(
            obs_dim=observation_dim,
            action_dim=action_dim,
            config=self.config,
            rngs=nnx.Rngs(0),
        )
        graph, state = nnx.split(policy)
        # Upstream checkpoints are pickles. Only explicitly supplied trusted
        # local files are accepted by the CLI; no remote loading occurs here.
        with Path(checkpoint).open("rb") as stream:
            loaded = pickle.load(stream)
        expected = state.to_pure_dict()
        if jax.tree.structure(expected) != jax.tree.structure(loaded):
            raise ValueError(
                "Checkpoint parameter structure does not match the native RTC model"
            )
        for want, found in zip(
            jax.tree.leaves(expected), jax.tree.leaves(loaded), strict=True
        ):
            if getattr(found, "shape", None) != want.shape:
                raise ValueError(
                    "Checkpoint parameter shape does not match the native RTC model"
                )
        loaded = jax.tree.map(jnp.asarray, loaded)
        if not all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(loaded)):
            raise ValueError("Checkpoint contains nonfinite parameters")
        state.replace_by_pure_dict(loaded)
        self.policy = nnx.merge(graph, state)
        self._compiled = {}
        self.metadata = {
            "backend": "jax_flax_rtc",
            "source_file": str(Path(module.__file__).resolve()),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "action_horizon": self.action_horizon,
            "parameter_count": sum(x.size for x in jax.tree.leaves(loaded)),
            "parameter_dtypes": sorted({str(x.dtype) for x in jax.tree.leaves(loaded)}),
            "checkpoint_structure_and_shapes_checked": True,
            "quantized_kernel": False,
        }

    def reset(self, seed):
        import jax

        self._rng = jax.random.fold_in(jax.random.key(seed), 2)
        self._requests = 0

    def infer(self, observation, flow_steps):
        import jax
        import numpy as np

        return np.asarray(jax.device_get(self.infer_device(observation, flow_steps)))

    def infer_device(self, observation, flow_steps):
        """Return the existing sampler output without an explicit host transfer."""
        import jax
        import jax.numpy as jnp

        if flow_steps not in self._compiled:
            self._compiled[flow_steps] = jax.jit(
                lambda key, obs: self.policy.action(key, obs[None], flow_steps)[0]
            )
        key = jax.random.fold_in(self._rng, self._requests)
        self._requests += 1
        return self._compiled[flow_steps](key, jnp.asarray(observation))
