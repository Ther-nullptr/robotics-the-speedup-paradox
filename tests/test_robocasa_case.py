"""Focused checks for RoboCasa episode identity and controller boundaries."""

from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from robotics_bench.protocols.static_runner import run_episode


def test_runner_reads_language_after_reset_once():
    events = []
    simulator = SimpleNamespace(description=None)

    def reset(*args, **kwargs):
        events.append("reset")
        simulator.description = "press the stop button on the microwave"
        return {}

    simulator.reset = reset
    simulator.step = lambda action: ({}, True, True)
    engine = SimpleNamespace(reset=lambda identity: None)

    def infer(obs, task, seed):
        events.append(task)
        return [[0] * 7]

    engine.infer_chunk = infer
    result = run_episode(
        engine,
        simulator,
        task="TurnOffMicrowave",
        init_state_id=0,
        env_seed=0,
        sampling_seed=195,
        max_steps=500,
        n_action_steps=1,
    )
    assert result["primitive_steps"] == 1
    assert events == ["reset", "press the stop button on the microwave"]


def test_mobile_action_mapping_requires_declared_controller_layout():
    from robotics_bench.simulators.robocasa import controller_action

    env = SimpleNamespace(
        action_dim=12,
        robots=[
            SimpleNamespace(
                composite_controller=SimpleNamespace(name="HYBRID_MOBILE_BASE"),
                _action_split_indexes={
                    "right": (0, 6),
                    "right_gripper": (6, 7),
                    "base": (8, 11),
                    "torso": (7, 8),
                },
            )
        ],
    )
    action = np.arange(7, dtype=float)
    actual = controller_action(env, action)
    np.testing.assert_array_equal(actual, np.concatenate([action, [0, 0, 0, 0, -1]]))
    env.robots[0]._action_split_indexes["right"] = (1, 7)
    with pytest.raises(ValueError, match="controller"):
        controller_action(env, action)


def test_reference_run_rejects_different_initial_observation(tmp_path):
    path = (
        Path(__file__).resolve().parents[1] / "benchmarks/static/cosmos_robocasa/run.py"
    )
    spec = importlib.util.spec_from_file_location("robocasa_case_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path / "initializations/000000"
    folder.mkdir(parents=True)
    expected = {"task": "TurnOffMicrowave", "initial_observation_sha256": "original"}
    (folder / "episode.json").write_text(json.dumps(expected))
    module.check_reference(tmp_path, 0, expected)
    with pytest.raises(RuntimeError, match="initial_observation_sha256"):
        module.check_reference(
            tmp_path, 0, {**expected, "initial_observation_sha256": "changed"}
        )
