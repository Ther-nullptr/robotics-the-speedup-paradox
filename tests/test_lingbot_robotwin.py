"""Critical LingBot cache, budget and action-coordinate behavior."""

from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest


@pytest.mark.parametrize(
    "success_at,expected_steps,expected_updates", [(5, 5, 0), (None, 18, 1)]
)
def test_sync_runner_preserves_conditioning_prefix_and_stops_inside_chunk(
    success_at, expected_steps, expected_updates
):
    from robotics_bench.protocols.lingbot_runner import run_episode

    executed, updates, frames = [], [], []
    simulator = SimpleNamespace(description="adjust the bottle", step_count=0)

    def step(action):
        executed.append(float(action[0]))
        simulator.step_count += 1
        return {"step": simulator.step_count}, simulator.step_count == success_at

    simulator.step = step
    simulator.render = lambda: simulator.step_count
    engine = SimpleNamespace(reset=lambda prompt: None, calls=0)

    def infer(obs, prompt):
        action = np.zeros((16, 2, 16), dtype=np.float32)
        action[0] = np.arange(32).reshape(2, 16) + engine.calls * 100
        engine.calls += 1
        return action

    engine.infer_chunk = infer
    engine.update_cache = lambda obs, state: updates.append((obs, state.copy()))
    result = run_episode(
        engine, simulator, {"step": 0}, max_steps=18, on_frame=frames.append
    )
    assert result["primitive_steps"] == expected_steps
    assert executed == (list(range(16, 32)) + [100, 101])[:expected_steps]
    assert len(updates) == expected_updates
    assert len(frames) == expected_steps + 1
    if updates:
        assert [o["step"] for o in updates[0][0]] == [4, 8, 12, 16]
        assert updates[0][1].shape == (16, 2, 16)


def test_relative_pose_composition_preserves_both_grippers():
    from robotics_bench.simulators.robotwin import absolute_eef_action

    initial = np.array([1, 2, 3, 0, 0, 0, 1, 0, 4, 5, 6, 0, 0, 0, 1, 0], dtype=float)
    relative = np.array([0.1, 0.2, 0.3, 0, 0, 0, 1, 0.8, -0.1, 0, 0, 0, 0, 0, 1, 0.4])
    expected = initial.copy()
    expected[:3] += relative[:3]
    expected[8:11] += relative[8:11]
    expected[[7, 15]] = relative[[7, 15]]
    np.testing.assert_allclose(absolute_eef_action(relative, initial), expected)
    relative[3:7] = 0
    with pytest.raises(ValueError, match="quaternion"):
        absolute_eef_action(relative, initial)


def test_text_encoder_restores_only_the_checkpoint_shared_embedding_alias():
    from robotics_bench.engines.lingbot_server import restore_text_embedding_alias

    shared = object()
    model = SimpleNamespace(
        shared=shared, encoder=SimpleNamespace(embed_tokens=object())
    )
    model.set_input_embeddings = lambda embedding: setattr(
        model.encoder, "embed_tokens", embedding
    )
    assert restore_text_embedding_alias(model, {"shared.weight"})
    assert model.encoder.embed_tokens is shared
    untied = object()
    model.encoder.embed_tokens = untied
    assert not restore_text_embedding_alias(
        model, {"shared.weight", "encoder.embed_tokens.weight"}
    )
    assert model.encoder.embed_tokens is untied


@pytest.mark.parametrize(
    "schedule,delay",
    [
        ("sync", 0),
        ("paper_async", 0),
        ("paper_async", 2),
        ("paper_async", 6),
        ("paper_async", 16),
    ],
)
def test_history_policy_delays_entire_cache_without_leaking_live_observations(
    schedule, delay
):
    from robotics_bench.protocols.lingbot_runner import run_episode

    observation = {"image": np.array([0]), "state": np.array([0])}
    simulator = SimpleNamespace(description="adjust the bottle", step_count=0)
    inputs, updates, events, executed, frames = [], [], [], [], []

    def step(action):
        executed.append(float(action[0]))
        simulator.step_count += 1
        # A simulator may reuse its buffers; previous snapshots must survive.
        for value in observation.values():
            value[:] = simulator.step_count
        return observation, False

    simulator.step = step
    simulator.render = lambda: simulator.step_count
    engine = SimpleNamespace(reset=lambda prompt: None)

    def infer(snapshot, prompt):
        inputs.append(deepcopy(snapshot))
        action = np.zeros((16, 2, 16), dtype=np.float32)
        action[0] = np.arange(32).reshape(2, 16) + (len(inputs) - 1) * 100
        # A consumer may mutate its input; history and simulator stay isolated.
        snapshot["image"][:] = -999
        return action

    def update(keyframes, action):
        updates.append((deepcopy(keyframes), action.copy()))
        for snapshot in keyframes:
            snapshot["state"][:] = -999

    engine.infer_chunk, engine.update_cache = infer, update
    result = run_episode(
        engine,
        simulator,
        observation,
        max_steps=50,
        schedule=schedule,
        overlap_actions=delay,
        on_event=events.append,
        on_frame=frames.append,
    )
    assert result["primitive_steps"] == 50
    assert result["cache_updates"] == 2  # No cache update for the truncated last chunk.
    assert executed == list(range(16, 32)) + list(range(100, 132)) + [200, 201]
    assert frames == list(range(51))  # Recording follows live control, not stale input.
    expected_inference_steps = [0, 16 - delay, 48 - delay]
    for snapshot, source_step in zip(inputs, expected_inference_steps, strict=True):
        assert snapshot["image"].tolist() == snapshot["state"].tolist() == [source_step]
    cache_events = [e for e in events if e["kind"] == "cache_update"]
    for (keyframes, action), nominal, event in zip(
        updates,
        [list(range(4, 17, 4)), list(range(20, 49, 4))],
        cache_events,
        strict=True,
    ):
        expected = [max(0, step - delay) for step in nominal]
        assert [int(o["image"][0]) for o in keyframes] == expected
        assert [int(o["state"][0]) for o in keyframes] == expected
        assert event["observation_steps"] == expected
        assert event["nominal_keyframe_steps"] == nominal
        assert max(expected) == event["control_step"] - delay
        # Retain native predicted-action conditioning, including first condition frame.
        base = 0 if nominal[0] == 4 else 100
        np.testing.assert_array_equal(action[0], np.arange(32).reshape(2, 16) + base)
    inference_events = [e for e in events if e["kind"] == "inference"]
    assert [e["observation_step"] for e in inference_events] == expected_inference_steps
    assert [
        e["cache_observation_max_step"] for e in inference_events
    ] == expected_inference_steps
    assert [e["history_offset_steps"] for e in inference_events] == [0, delay, delay]


@pytest.mark.parametrize(
    "schedule,delay",
    [
        ("sync", 2),
        ("paper_async", -1),
        ("paper_async", 17),
        ("paper_async", True),
        ("async", 2),
    ],
)
def test_invalid_history_policy_fails_before_model_or_environment_use(schedule, delay):
    from robotics_bench.protocols.lingbot_runner import run_episode

    with pytest.raises(ValueError, match="schedule|overlap_actions"):
        run_episode(
            None, None, {}, max_steps=10, schedule=schedule, overlap_actions=delay
        )
