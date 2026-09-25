"""CPU tests for inference timing boundaries and artificial release delays."""

from contextlib import nullcontext
from queue import Queue
from threading import Event
from types import SimpleNamespace
import sys

import pytest

from robotics_bench.engines.dynamicvla import DynamicVLAEngine
from robotics_bench.models.dynamicvla import streaming


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf")])
def test_invalid_delay_is_rejected(tmp_path, delay):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        DynamicVLAEngine(tmp_path, streaming=True, extra_delay_ms=delay)


@pytest.mark.parametrize(
    "options",
    [
        dict(measure_inference=True),
        dict(extra_delay_ms=1),
        dict(episode_seed_mode=True, seed=42),
    ],
)
def test_latency_study_requires_streaming(tmp_path, options):
    with pytest.raises(ValueError, match="require native streaming"):
        DynamicVLAEngine(tmp_path, **options)


def test_episode_rng_requires_base_seed_and_is_opt_in(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="explicit base seed"):
        DynamicVLAEngine(tmp_path, streaming=True, episode_seed_mode=True)
    seeds = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=seeds.append))
    streaming.seed_episode(SimpleNamespace(runtime_seed=10), 5)
    streaming.seed_episode(SimpleNamespace(runtime_seed=10, episode_seed_mode=True), 5)
    assert seeds == [15]


@pytest.mark.parametrize("measure,delay", [(False, 0), (True, 0), (True, 50)])
def test_worker_timing_excludes_native_pacing_and_release_delay(
    monkeypatch, measure, delay
):
    clock = [100.0]
    syncs = []
    seeds = []
    sleeps = []
    stop = Event()

    class Tensor:
        def to(self, device):
            clock[0] += 0.003  # Host-to-device service.
            return self

        def __getitem__(self, item):
            return self

        def transpose(self, *args):
            return self

        def cpu(self):
            clock[0] += 0.002  # Complete device-to-host output.
            return self

    class Policy:
        def __init__(self):
            self.generation_events = []
            self._generated_chunks = 0

        def eval(self):
            return self

        def to(self, device):
            return self

        def reset(self):
            self.generation_events.clear()

        def parameters(self):
            return iter([SimpleNamespace(dtype="torch.float32")])

        def _prepare_batch(self, batch):
            return batch

        def _get_action_chunk(self, batch, noise=None):
            if "index" not in batch:  # Unmeasured startup warmup.
                return Tensor()
            clock[0] += 0.010 + 0.020
            self.generation_events.append(
                dict(
                    kind="chunk_generated",
                    chunk_id=0,
                    native_pacing_ms=20.0,
                    chunk_observation_index=batch["index"],
                    chunk_observation_sim_time_s=1.0,
                    chunk_observation_wall_s=99.0,
                )
            )
            return Tensor()

    class Events(Queue):
        def put(self, event, *args, **kwargs):
            if event["kind"] == "chunk_started":
                assert reads[0] == 0  # Progress publication precedes service timing.
            super().put(event, *args, **kwargs)
            if event["kind"] in ("chunk_generated", "worker_error"):
                stop.set()

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    reads = [0]

    def perf_counter():
        reads[0] += 1
        # Account for OS preemption both before and inside the release gate,
        # including runs with zero requested delay and no intentional sleep.
        if reads[0] == 3:
            clock[0] += 0.004
        elif reads[0] == 4:
            clock[0] += 0.002
        return clock[0]

    monkeypatch.setattr(streaming.time, "perf_counter", perf_counter)
    monkeypatch.setattr(streaming.time, "sleep", sleep)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Tensor=Tensor,
            manual_seed=seeds.append,
            no_grad=nullcontext,
            cuda=SimpleNamespace(synchronize=syncs.append),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "robotics_bench.models.dynamicvla.modeling_dynamicvla",
        SimpleNamespace(
            DynamicVLAPolicy=SimpleNamespace(from_pretrained=lambda *a, **k: Policy())
        ),
    )
    config = SimpleNamespace(
        device="cuda:2",
        input_features={},
        runtime_seed=42,
        episode_seed_mode=True,
        use_delta_action=False,
        measure_inference=measure,
        extra_delay_ms=delay,
    )
    batch = {"_episode_id": 3, "index": 7, "observation.state": Tensor()}
    results, events = Queue(), Events()
    streaming.worker(
        "checkpoint", config, {"obs": (batch, None)}, results, events, stop
    )
    assert results.get_nowait() == {"initialized": True}
    result = results.get_nowait()
    if measure:
        assert events.get_nowait() == {
            "kind": "chunk_started",
            "episode_id": 3,
            "chunk_id": 0,
            "observation_index": 7,
        }
    event = events.get_nowait()
    assert events.empty()  # Default native runs emit only the completed event.
    assert event["kind"] == "chunk_generated"
    assert event["model_parameter_dtypes"] == (["torch.float32"] if measure else None)
    assert (
        event["worker_compute_ms"] == pytest.approx(15)
        if measure
        else event["worker_compute_ms"] is None
    )
    assert event["extra_delay_requested_ms"] == delay
    assert event["extra_delay_actual_ms"] == pytest.approx(delay + 2)
    assert event["post_compute_overhead_ms"] == pytest.approx(4)
    assert event["delay_started_wall_s"] == pytest.approx(100.039)
    assert event["compute_ready_wall_s"] == pytest.approx(100.035)
    assert event["released_wall_s"] == pytest.approx(100.041 + delay / 1000)
    assert (
        event["released_wall_s"] - event["compute_ready_wall_s"]
    ) * 1000 == pytest.approx(
        event["post_compute_overhead_ms"] + event["extra_delay_actual_ms"]
    )
    assert event["result_ready_wall_s"] == event["released_wall_s"]
    assert result["chunk_observation_index"] == 7
    assert result["chunk_observation_sim_time_s"] == 1.0
    assert result["chunk_observation_wall_s"] == 99.0
    assert sleeps == ([delay / 1000] if delay else [])
    assert syncs == (["cuda:2"] if measure else [])
    assert seeds == [42, 45]


def owned_policy_method(name, namespace):
    """Exercise the owned method without installing its optional ML dependencies."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "src/robotics_bench/models/dynamicvla/modeling_dynamicvla.py"
    )
    tree = ast.parse(source.read_text())
    policy = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DynamicVLAPolicy"
    )
    method = next(
        node
        for node in policy.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("measure", [False, True])
def test_generation_sync_and_native_pacing_have_distinct_boundaries(measure):
    clock = [0.0]
    syncs = []
    sleeps = []

    class Tensor:
        dtype = "torch.float32"
        shape = (1, 20, 7)
        device = SimpleNamespace(type="cuda")

        def __getitem__(self, key):
            return self

    def sample(*args, **kwargs):
        clock[0] += 0.010
        return Tensor()

    def unnormalize(value):
        clock[0] += 0.001
        return value

    def sync(device):
        syncs.append(device)
        clock[0] += 0.006

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    method = owned_policy_method(
        "_get_action_chunk",
        dict(
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=sync)),
            time=SimpleNamespace(perf_counter=lambda: clock[0], sleep=sleep),
            logging=SimpleNamespace(info=lambda *args: None),
            ACTION="action",
            OBS_STATE="observation.state",
        ),
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(
            measure_inference=measure,
            action_feature=SimpleNamespace(shape=[7]),
            adapt_to_pi_aloha=False,
        ),
        _queues={},
        prepare_images=lambda batch: (None, None),
        prepare_state=lambda batch: None,
        prepare_language=lambda batch: (None, None),
        model=SimpleNamespace(sample_actions=sample),
        unnormalize_outputs=unnormalize,
        _generated_chunks=0,
        _episode_id=2,
        generation_events=[],
    )
    method(policy, {"observation.state": Tensor(), "index": 4, "dt_scale": 3})
    event = policy.generation_events[0]
    assert event["model_host_duration_s"] == pytest.approx(0.011)
    assert event["native_pacing_ms"] == pytest.approx(22)
    assert sleeps == pytest.approx([0.022])
    assert len(syncs) == (2 if measure else 0)
    if measure:
        assert event["model_compute_ms"] == pytest.approx(17)
    else:
        assert event["model_compute_ms"] is None
    assert event["action_dtype"] == "torch.float32"


def test_native_action_queue_retains_generating_observation(monkeypatch):
    from collections import deque
    from queue import Empty

    class Actions(list):
        def size(self, dimension):
            return len(self)

    monkeypatch.setattr(streaming, "check_worker", lambda policy: None)
    method = owned_policy_method(
        "_get_streaming_action",
        dict(
            __package__="robotics_bench.models.dynamicvla",
            Empty=Empty,
            ACTION="action",
            logging=SimpleNamespace(debug=lambda *args: None),
        ),
    )
    output = Queue()
    output.put(
        {
            "actions": Actions(["a", "b", "c", "d"]),
            "index": 10,
            "episode_id": 2,
            "chunk_id": 3,
            "chunk_observation_index": 10,
            "chunk_observation_sim_time_s": 0.5,
            "chunk_observation_wall_s": 100.0,
        }
    )
    policy = SimpleNamespace(
        q_in={},
        q_out=output,
        config=SimpleNamespace(n_action_steps=4),
        generation_events=[],
        _queues={"action": deque(maxlen=4)},
        _action_index=0,
    )
    assert method(policy, {"index": 11, "_episode_id": 2}) == "b"
    assert policy.last_action_metadata == {
        "chunk_id": 3,
        "action_index": 11,
        "chunk_observation_index": 10,
        "chunk_observation_sim_time_s": 0.5,
        "chunk_observation_wall_s": 100.0,
    }
    # A later queue pop must still identify observation 10, not selection frame 12.
    assert method(policy, {"index": 12, "_episode_id": 2}) == "c"
    assert policy.last_action_metadata["chunk_observation_index"] == 10
    assert policy.last_action_metadata["action_index"] == 12
