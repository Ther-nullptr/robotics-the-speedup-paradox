"""CPU-only static rollout tests with independent action and observation traces."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/robotics_bench/protocols/static_runner.py"
)


def api():
    assert SOURCE.is_file(), "static_runner.py is not implemented"
    spec = importlib.util.spec_from_file_location("tested_static_runner", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSimulator:
    task_name = "task_a"
    description = "task_a"
    control_dt = 0.05

    def __init__(self, *, success_at=None, terminal_at=None, fail_at=None):
        self.success_at = success_at
        self.terminal_at = terminal_at
        self.fail_at = fail_at
        self.actions = []
        self.resets = []
        self.steps = 0
        self.observation = {
            "primary_image": [0],
            "wrist_image": [100],
            "proprio": [0] * 9,
        }

    def _update(self):
        self.observation["primary_image"][0] = self.steps
        self.observation["wrist_image"][0] = self.steps + 100
        self.observation["proprio"][:] = [self.steps] * 9
        return self.observation

    def reset(self, initial, seed=None):
        self.resets.append((initial, seed))
        self.steps = 0
        return self._update()

    def step(self, action):
        if self.fail_at == self.steps + 1:
            raise RuntimeError("simulator failed")
        self.actions.append(action)
        self.steps += 1
        return (
            self._update(),
            self.steps == self.success_at,
            self.steps == self.success_at or self.steps == self.terminal_at,
        )

    def render(self):
        return self.steps


class FakeEngine:
    def __init__(self, horizon=5, *, fail=False, mutate_input=False):
        self.horizon = horizon
        self.fail = fail
        self.mutate_input = mutate_input
        self.requests = []
        self.resets = []

    def reset(self, episode_id):
        self.resets.append(episode_id)

    def infer_chunk(self, observation, task, sampling_seed):
        self.requests.append((deepcopy(observation), task, sampling_seed))
        if self.fail:
            raise RuntimeError("inference failed")
        if self.mutate_input:
            observation["proprio"][0] = -999
        base = 10 * len(self.requests)
        return [[base + index] * 7 for index in range(self.horizon)]


def run(engine=None, simulator=None, **kwargs):
    defaults = {
        "task": "task_a",
        "init_state_id": 3,
        "env_seed": 42,
        "sampling_seed": 7,
        "max_steps": 5,
        "n_action_steps": 2,
    }
    return api().run_episode(
        engine or FakeEngine(), simulator or FakeSimulator(), **(defaults | kwargs)
    )


def test_only_prefix_actions_are_executed_and_budget_counts_actual_steps():
    engine, simulator = FakeEngine(horizon=5), FakeSimulator()
    result = run(engine, simulator)
    assert [action[0] for action in simulator.actions] == [10, 11, 20, 21, 30]
    assert result == {
        "task": "task_a",
        "init_state_id": 3,
        "env_seed": 42,
        "success": False,
        "primitive_steps": 5,
        "max_primitive_steps": 5,
        "inference_calls": 3,
        "termination_reason": "budget_exhausted",
    }
    assert simulator.resets == [(3, 42)]
    assert engine.resets == [("task_a", 3, 42)]
    assert [request[2] for request in engine.requests] == [7, 7, 7]


@pytest.mark.parametrize("overlap,expected", [(0, [0, 2, 4]), (2, [0, 0, 2])])
def test_history_is_measured_in_primitive_steps_and_state_matches_images(
    overlap, expected
):
    engine = FakeEngine()
    requests = []
    run(
        engine,
        schedule="paper_async",
        overlap_actions=overlap,
        on_request=requests.append,
    )
    for (observation, _, _), tick in zip(engine.requests, expected, strict=True):
        assert observation["primary_image"] == [tick]
        assert observation["wrist_image"] == [tick + 100]
        assert observation["proprio"] == [tick] * 9
    assert [row["control_step"] for row in requests] == [0, 2, 4]
    assert [row["observation_step"] for row in requests] == expected
    assert [row["history_offset_steps"] for row in requests] == [
        control - observation for control, observation in zip([0, 2, 4], expected)
    ]
    assert [row["inference_index"] for row in requests] == [1, 2, 3]
    assert not any("paper" in key for row in requests for key in row)


def test_history_warmup_uses_current_snapshot_until_full_window():
    engine = FakeEngine()
    run(
        engine, n_action_steps=1, schedule="paper_async", overlap_actions=1, max_steps=3
    )
    assert [request[0]["primary_image"][0] for request in engine.requests] == [0, 0, 1]


def test_engine_cannot_mutate_history_or_simulator_owned_buffers():
    engine, simulator = FakeEngine(mutate_input=True), FakeSimulator()
    run(engine, simulator, schedule="paper_async", overlap_actions=2)
    assert [request[0]["proprio"][0] for request in engine.requests] == [0, 0, 2]
    assert simulator.observation["proprio"] == [5] * 9


def test_success_at_last_budget_step_wins_over_timeout_and_includes_terminal_frame():
    frames = []
    simulator = FakeSimulator(success_at=5)
    result = run(simulator=simulator, on_frame=frames.append)
    assert result["success"] is True
    assert result["termination_reason"] == "success"
    assert result["primitive_steps"] == 5
    assert frames == [0, 1, 2, 3, 4, 5]


def test_early_terminal_stops_steps_and_preserves_initial_and_final_frame():
    frames = []
    simulator = FakeSimulator(terminal_at=1)
    result = run(simulator=simulator, on_frame=frames.append)
    assert result["success"] is False
    assert result["termination_reason"] == "environment_terminated"
    assert result["primitive_steps"] == 1
    assert len(simulator.actions) == 1
    assert frames == [0, 1]


def test_repeated_episodes_reset_queue_and_history():
    engine, simulator = FakeEngine(), FakeSimulator(success_at=1)
    run(engine, simulator, schedule="paper_async", overlap_actions=2)
    run(engine, simulator, init_state_id=4, schedule="paper_async", overlap_actions=2)
    assert [action[0] for action in simulator.actions] == [10, 20]
    assert [request[0]["primary_image"][0] for request in engine.requests] == [0, 0]
    assert simulator.resets == [(3, 42), (4, 42)]
    assert engine.resets == [("task_a", 3, 42), ("task_a", 4, 42)]


def test_engine_receives_language_while_result_keeps_stable_task_name():
    engine, simulator = FakeEngine(), FakeSimulator()
    simulator.description = "Move the object onto the tray."
    result = run(engine, simulator, max_steps=1)
    assert engine.requests[0][1] == simulator.description
    assert result["task"] == "task_a"


def test_inference_errors_propagate_without_executing_actions():
    simulator = FakeSimulator()
    with pytest.raises(RuntimeError, match="inference failed"):
        run(FakeEngine(fail=True), simulator)
    assert simulator.actions == []


def test_step_errors_propagate_without_becoming_task_failures():
    simulator = FakeSimulator(fail_at=2)
    with pytest.raises(RuntimeError, match="simulator failed"):
        run(simulator=simulator)
    assert len(simulator.actions) == 1


@pytest.mark.parametrize("callback", ["on_frame", "on_request"])
def test_callback_errors_propagate(callback):
    def fail(value):
        raise OSError("artifact failed")

    with pytest.raises(OSError, match="artifact failed"):
        run(**{callback: fail})


@pytest.mark.parametrize("horizon", [0, 1])
def test_short_or_empty_chunk_is_rejected_before_environment_step(horizon):
    simulator = FakeSimulator()
    with pytest.raises(ValueError, match="chunk"):
        run(FakeEngine(horizon=horizon), simulator)
    assert simulator.actions == []


@pytest.mark.parametrize(
    "changes",
    [
        {"max_steps": 0},
        {"max_steps": True},
        {"n_action_steps": 0},
        {"n_action_steps": 1.5},
        {"overlap_actions": True},
        {"overlap_actions": -1},
        {"overlap_actions": 3, "schedule": "paper_async"},
        {"overlap_actions": 1, "schedule": "sync"},
        {"schedule": "async"},
        {"init_state_id": -1},
        {"env_seed": True},
        {"sampling_seed": -1},
        {"task": ""},
    ],
)
def test_invalid_protocol_rejected_before_reset(changes):
    simulator = FakeSimulator()
    with pytest.raises(ValueError):
        run(simulator=simulator, **changes)
    assert simulator.resets == []


def test_import_needs_no_third_party_packages():
    assert SOURCE.is_file(), "static_runner.py is not implemented"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import runpy,sys;runpy.run_path(sys.argv[1])",
            str(SOURCE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
