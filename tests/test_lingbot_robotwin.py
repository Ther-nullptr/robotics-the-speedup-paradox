"""Critical LingBot cache, budget and action-coordinate behavior."""

from types import SimpleNamespace

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
