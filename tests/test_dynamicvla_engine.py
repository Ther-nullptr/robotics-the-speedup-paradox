"""CPU contracts for episode boundaries, observations and chunk accounting."""

from collections import deque
from queue import Queue
from types import SimpleNamespace
import sys

import pytest

from robotics_bench.engines.dynamicvla import DynamicVLAEngine
from robotics_bench.models.dynamicvla.streaming import reset_episode


def fake_policy():
    class Policy:
        def __init__(self):
            self.generation_events = []
            self.config = SimpleNamespace(input_features={"observation.state": None})
            self.received = []

        def reset(self):
            self.generation_events = []

        def select_action(self, batch):
            self.received.append(batch)
            return None

    return Policy()


def test_engine_rejects_missing_checkpoint_without_loading_dependencies(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        DynamicVLAEngine(tmp_path).load()
    with pytest.raises(ValueError, match="positive integer"):
        DynamicVLAEngine(tmp_path, num_steps=0)


def test_initial_three_observations_history_and_episode_reset(monkeypatch, tmp_path):
    def transform(observations, *args):
        return {"indices": [o["index"] for o in observations]}

    monkeypatch.setitem(
        sys.modules,
        "robotics_bench.models.dynamicvla.preprocessing",
        SimpleNamespace(transform_observations=transform),
    )
    monkeypatch.setitem(
        sys.modules,
        "robotics_bench.models.dynamicvla.rotations",
        SimpleNamespace(get_quaternion=None),
    )
    engine = DynamicVLAEngine(tmp_path, device="cpu")
    engine.policy = fake_policy()
    engine._history = deque(maxlen=5)
    engine._offsets = [-4, 0]
    engine.rotation = "quat"
    engine.reset(episode_id=7)
    for index in range(4):
        assert (
            engine.select_action(
                {"episode_id": 7, "index": index, "observation.state": {}}
            )
            is None
        )
    assert engine.policy.received == [{"indices": [0, 3], "_episode_id": 7}]
    with pytest.raises(ValueError, match="increase"):
        engine.select_action({"episode_id": 7, "index": 3, "observation.state": {}})
    with pytest.raises(ValueError, match="episode_id"):
        engine.select_action({"episode_id": 6, "index": 4, "observation.state": {}})
    engine.reset(episode_id=8)
    assert not engine._history
    assert engine._valid_observations == 0


def test_actual_generation_count_is_not_selection_count(tmp_path):
    engine = DynamicVLAEngine(tmp_path)
    engine.policy = fake_policy()
    engine.policy.generation_events = [
        {"kind": "chunk_generated", "episode_id": 0, "chunk_id": 0},
        {"kind": "chunk_generated", "episode_id": 0, "chunk_id": 1},
    ]
    engine._action_calls = 20
    assert len(engine.drain_events()) == 2
    assert engine.drain_events() == []
    assert engine._counts == {"0": 2}
    engine._history = deque()
    engine.policy.generation_events = [
        {"kind": "chunk_generated", "episode_id": 0, "chunk_id": 2}
    ]
    engine.reset(episode_id=1)
    assert engine.drain_events()[0]["episode_id"] == 0
    assert engine._counts == {"0": 3}


def test_reset_barrier_drains_old_generation_before_ack():
    events = Queue()
    events.put({"kind": "chunk_generated", "episode_id": 3, "chunk_id": 5})
    events.put({"kind": "reset_ack", "episode_id": 4})
    policy = SimpleNamespace(
        q_in={"obs": "old"},
        q_events=events,
        worker=SimpleNamespace(is_alive=lambda: True),
        _stop=SimpleNamespace(is_set=lambda: False),
    )
    assert reset_episode(policy, 4) == [
        {"kind": "chunk_generated", "episode_id": 3, "chunk_id": 5}
    ]
    assert policy.q_in == {"reset": 4}


def test_reset_barrier_propagates_worker_error():
    events = Queue()
    events.put({"kind": "worker_error", "error": "checkpoint load failed"})
    policy = SimpleNamespace(
        q_in={},
        q_events=events,
        worker=SimpleNamespace(is_alive=lambda: True),
        _stop=SimpleNamespace(is_set=lambda: False),
    )
    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        reset_episode(policy, 0)


def test_dom_rotation_is_explicit_euler_by_default(tmp_path):
    assert DynamicVLAEngine(tmp_path).rotation == "euler"
    assert DynamicVLAEngine(tmp_path, rotation="rotvec").rotation == "rotvec"
    assert DynamicVLAEngine(tmp_path, rotation="quat").describe()["rotation"] == "quat"
    with pytest.raises(ValueError, match="rotation"):
        DynamicVLAEngine(tmp_path, rotation="guessed")


def test_indexed_device_load_validates_backend_then_places_on_requested_device(
    monkeypatch, tmp_path
):
    import json
    from dataclasses import dataclass

    @dataclass
    class Config:
        input_features: dict
        output_features: dict
        device: str
        enable_streaming: bool
        n_obs_steps: int = 1
        num_steps: int = 10

        def __post_init__(self):
            assert self.device in ("cuda", "cpu", "mps")

        @property
        def action_feature(self):
            return self.output_features["action"]

    class Model:
        @staticmethod
        def from_pretrained(path, config, local_files_only):
            assert local_files_only
            assert config.device == "cuda:2"
            policy = fake_policy()
            policy.config = config
            policy.eval = lambda: policy
            policy.to = lambda device: policy
            return policy

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(device=lambda name: SimpleNamespace(type=name.split(":")[0])),
    )
    monkeypatch.setitem(
        sys.modules,
        "lerobot.configs.types",
        SimpleNamespace(
            FeatureType=lambda x: x,
            NormalizationMode=lambda x: x,
            PolicyFeature=lambda **kw: SimpleNamespace(**kw),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "robotics_bench.models.dynamicvla.configuration_dynamicvla",
        SimpleNamespace(DynamicVLAConfig=Config),
    )
    monkeypatch.setitem(
        sys.modules,
        "robotics_bench.models.dynamicvla.modeling_dynamicvla",
        SimpleNamespace(DynamicVLAPolicy=Model),
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "dynamicvla",
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [7]}
                },
                "output_features": {"action": {"type": "ACTION", "shape": [7]}},
            }
        )
    )
    (tmp_path / "model.safetensors").touch()
    engine = DynamicVLAEngine(tmp_path, device="cuda:2").load()
    assert engine.policy.config.device == "cuda:2"
    assert engine.rotation == "euler"


@pytest.mark.parametrize("seed", [None, 42])
def test_worker_seeds_before_model_load_and_warmup(monkeypatch, seed):
    from contextlib import nullcontext
    from threading import Event
    from robotics_bench.models.dynamicvla.streaming import worker

    calls = []

    class Policy:
        generation_events = []

        def eval(self):
            return self

        def to(self, device):
            return self

        def _get_action_chunk(self, batch):
            calls.append("warmup")
            self.generation_events.append({"warmup": True})

    def load(*args, **kwargs):
        calls.append("load")
        return Policy()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda value: calls.append(("seed", value)), no_grad=nullcontext
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "robotics_bench.models.dynamicvla.modeling_dynamicvla",
        SimpleNamespace(DynamicVLAPolicy=SimpleNamespace(from_pretrained=load)),
    )
    stop = Event()
    stop.set()
    results, events = Queue(), Queue()
    worker(
        "local-checkpoint",
        SimpleNamespace(runtime_seed=seed, device="cpu", input_features={}),
        {},
        results,
        events,
        stop,
    )
    expected = ([] if seed is None else [("seed", seed)]) + ["load", "warmup"]
    assert calls == expected
    assert results.get_nowait() == {"initialized": True}
    assert events.empty()
