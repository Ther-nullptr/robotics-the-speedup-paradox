"""Offline DynamicVLA inference engine for its native DOM case.

Heavy dependencies are imported only by load/select_action. Source model and
preprocessing are maintained in this package; external source trees are unused.
"""

from collections import deque
import copy
from dataclasses import fields
import json
from pathlib import Path


class DynamicVLAEngine:
    def __init__(
        self,
        checkpoint,
        streaming=False,
        device="cuda:0",
        num_steps=None,
        rotation="euler",
        seed=None,
    ):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if rotation not in ("quat", "rotvec", "euler"):
            raise ValueError("Unsupported action rotation representation")
        self.seed = seed
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer")
        self.rotation = rotation
        self.streaming = bool(streaming)
        self.device = str(device)
        if num_steps is not None and (type(num_steps) is not int or num_steps < 1):
            raise ValueError("num_steps must be a positive integer")
        self.num_steps = num_steps
        self.policy = None
        self.episode_id = -1
        self._events = []
        self._counts = {}
        self._action_calls = 0

    def load(self):
        if self.policy is not None:
            return self
        config_path = self.checkpoint / "config.json"
        if (
            not config_path.is_file()
            or not (self.checkpoint / "model.safetensors").is_file()
        ):
            raise FileNotFoundError(
                "checkpoint must contain config.json and model.safetensors"
            )
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from robotics_bench.models.dynamicvla.configuration_dynamicvla import (
            DynamicVLAConfig,
        )
        from robotics_bench.models.dynamicvla.modeling_dynamicvla import (
            DynamicVLAPolicy,
        )

        raw = json.loads(config_path.read_text())
        if raw.get("type") != "dynamicvla":
            raise ValueError("Expected a DynamicVLA checkpoint")
        kwargs = {
            k: v
            for k, v in raw.items()
            if k in {f.name for f in fields(DynamicVLAConfig)} and v is not None
        }
        for name in ("input_features", "output_features"):
            kwargs[name] = {
                k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                for k, v in raw[name].items()
            }
        if "normalization_mapping" in kwargs:
            kwargs["normalization_mapping"] = {
                k: NormalizationMode(v)
                for k, v in kwargs["normalization_mapping"].items()
            }
        # LeRobot 0.3 validates only backend names, not indexed Torch devices.
        device_type = torch.device(self.device).type
        kwargs.update(device=device_type, enable_streaming=self.streaming)
        if self.num_steps is not None:
            kwargs["num_steps"] = self.num_steps
        config = DynamicVLAConfig(**kwargs)
        if config.device != device_type:
            raise RuntimeError(f"Requested device {self.device} is unavailable")
        config.device = self.device
        config.runtime_seed = self.seed
        self._offsets = (raw.get("delta_timestamps") or {}).get("observation", [0])
        if len(self._offsets) != config.n_obs_steps or any(
            type(x) is not int or x > 0 for x in self._offsets
        ):
            raise ValueError(
                "Observation timestamps must be nonpositive integer indices matching n_obs_steps"
            )
        action_dim = config.action_feature.shape[0]
        expected_action_dim = 8 if self.rotation == "quat" else 7
        if action_dim != expected_action_dim:
            raise ValueError(
                f"{self.rotation} rotation requires action dimension {expected_action_dim}, got {action_dim}"
            )
        if raw.get("rotation") not in (None, self.rotation):
            raise ValueError("Requested rotation conflicts with checkpoint metadata")
        self._history = deque(maxlen=1 - min(self._offsets))
        if self.streaming:
            if config.n_action_steps != config.chunk_size:
                raise ValueError(
                    "Native streaming requires n_action_steps == chunk_size"
                )
            self.policy = DynamicVLAPolicy.get_streaming_model(
                str(self.checkpoint), config
            )
        else:
            self.policy = (
                DynamicVLAPolicy.from_pretrained(
                    str(self.checkpoint), config=config, local_files_only=True
                )
                .eval()
                .to(self.device)
            )
        self.reset(episode_id=0)
        return self

    def _record_events(self, events):
        for event in events:
            if event["kind"] == "chunk_generated":
                episode = str(event["episode_id"])
                self._counts[episode] = self._counts.get(episode, 0) + 1
            self._events.append(event)

    def drain_events(self):
        if self.policy is not None:
            from robotics_bench.models.dynamicvla.streaming import drain

            self._record_events(drain(self.policy))
        result, self._events = self._events, []
        return result

    def reset(self, episode_id=None):
        if self.policy is None:
            raise RuntimeError("Call load() before reset()")
        if episode_id is None:
            episode_id = self.episode_id + 1
        if type(episode_id) is not int or episode_id < 0:
            raise ValueError("episode_id must be a nonnegative integer")
        from robotics_bench.models.dynamicvla.streaming import drain, reset_episode

        self._record_events(drain(self.policy))
        if self.streaming:
            self._record_events(reset_episode(self.policy, episode_id))
        self.policy.reset()
        self.policy._episode_id = episode_id
        self.episode_id = episode_id
        self._history.clear()
        self._valid_observations = 0
        self._action_calls = 0

    def select_action(self, observation):
        if self.policy is None:
            raise RuntimeError("Call load() before select_action()")
        import numpy as np
        from robotics_bench.models.dynamicvla.preprocessing import (
            transform_observations,
        )
        from robotics_bench.models.dynamicvla.rotations import get_quaternion

        if "episode_id" in observation and observation["episode_id"] != self.episode_id:
            raise ValueError("Observation episode_id does not match engine reset")
        for key in self.policy.config.input_features:
            if key not in observation:
                raise ValueError(f"Missing observation feature: {key}")
        if self._history and observation["index"] <= self._history[-1]["index"]:
            raise ValueError("Observation indices must increase within an episode")
        self._history.append(copy.deepcopy(observation))
        self._valid_observations += 1
        if self._valid_observations <= 3:
            return None
        selected = [
            self._history[max(0, len(self._history) - 1 + offset)]
            for offset in self._offsets
        ]
        batch = transform_observations(
            selected,
            self.rotation,
            self.policy.config.input_features,
            "cpu" if self.streaming else self.device,
        )
        batch["_episode_id"] = self.episode_id
        action = self.policy.select_action(batch)
        self._action_calls += 1
        if action is None:
            return None
        action = action.detach().cpu().numpy()
        result = np.concatenate(
            [
                action[:, :3],
                get_quaternion(action[:, 3:-1], self.rotation),
                action[:, -1:],
            ],
            axis=-1,
        )
        if result.shape != (1, 8) or not np.isfinite(result).all():
            raise ValueError(
                "DynamicVLA must produce a finite [1, 8] absolute pose/gripper action"
            )
        return result

    @property
    def last_action_metadata(self):
        return copy.deepcopy(getattr(self.policy, "last_action_metadata", None))

    def describe(self):
        result = {
            "engine": "dynamicvla",
            "checkpoint": str(self.checkpoint),
            "streaming": self.streaming,
            "rotation": self.rotation,
            "seed": self.seed,
            "rng_policy": "Seed once before model construction; continuous RNG across episodes; native streaming warmup consumes RNG",
            "device": self.device,
            "episode_id": self.episode_id,
            "generated_chunks_by_episode": dict(self._counts),
            "action_selection_calls": self._action_calls,
            "initial_dummy_observations": 3,
            "source": "robotics_bench.models.dynamicvla",
        }
        if self.policy is not None:
            result.update(
                n_obs_steps=self.policy.config.n_obs_steps,
                chunk_size=self.policy.config.chunk_size,
                n_action_steps=self.policy.config.n_action_steps,
                num_steps=self.policy.config.num_steps,
                rotation=self.rotation,
                use_delta_action=self.policy.config.use_delta_action,
            )
        return result

    def finish_episode(self):
        """Wait for an in-flight generation and collect its evidence at terminal."""
        if self.policy is not None and self.streaming:
            from robotics_bench.models.dynamicvla.streaming import reset_episode

            self._record_events(reset_episode(self.policy, self.episode_id))
        return self.drain_events()

    def close(self):
        if self.policy is not None:
            from robotics_bench.models.dynamicvla.streaming import close

            try:
                # Retain events for a subsequent drain_events() call after close.
                self._events.extend(self.finish_episode())
            finally:
                close(self.policy)
                self.policy = None
