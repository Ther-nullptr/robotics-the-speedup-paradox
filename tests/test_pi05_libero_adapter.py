"""CPU checks for explicit LIBERO initial-state selection and terminal handling."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/static/pi05_libero/libero_adapter.py"
)


def api():
    assert ADAPTER.is_file(), "libero_adapter.py is not implemented"
    spec = importlib.util.spec_from_file_location("libero_adapter_tested", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RawEnv:
    def __init__(self):
        self.events = []
        self.state = None

    def seed(self, seed):
        self.events.append(("seed", seed))

    def reset(self):
        self.events.append(("reset",))
        self.state = -1
        return {"state": self.state}

    def set_init_state(self, state):
        self.events.append(("set_init_state", state))
        self.state = state
        return {"state": state}

    def step(self, action):
        self.events.append(("step", action))
        if action == "raise":
            raise RuntimeError("step failed")
        return {"state": self.state}, 1.0, action == "success", {}


class FakeBase:
    def __init__(self, episode_index=0, num_steps_wait=2):
        self.task = "move_object"
        self.task_id = 0
        self.init_states = True
        self._init_states = list(range(50))
        self._init_state_id = episode_index
        self.num_steps_wait = num_steps_wait
        self._env = RawEnv()
        # Actual LeRobot constructor resets the low-level env, not self.reset.
        self._env.reset()

    def _format_raw_obs(self, raw):
        return {"formatted": raw["state"]}

    def step(self, action):
        raw, reward, done, info = self._env.step(action)
        obs = self._format_raw_obs(raw)
        info["is_success"] = done
        if done:
            info["final_info"] = {"is_success": True}
            self.reset()
        return obs, reward, done, False, info


def gym_reset(env, *, seed=None):
    env._env.events.append(("gym_reset", seed))


def environment(**kwargs):
    cls = api().build_adapter_class(FakeBase, gym_reset, lambda: "dummy")
    return cls(**kwargs)


def test_import_requires_no_model_or_simulator_packages():
    assert ADAPTER.is_file(), "libero_adapter.py is not implemented"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1]); "
            "assert not any(x in sys.modules for x in ('torch','gymnasium','lerobot','libero'))",
            str(ADAPTER),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fifty_explicit_resets_select_each_initial_state_once_despite_success():
    env = environment(episode_ids=list(range(50)))
    ids = []
    for episode in range(50):
        obs, info = env.reset(seed=42)
        ids.append(env.evaluation_episode_id)
        assert obs == {"formatted": episode}
        assert info["evaluation_episode_id"] == episode
        terminal = env.step("success")
        assert terminal[2] is True
        assert terminal[4]["is_success"] is True
        assert terminal[4]["final_info"]["is_success"] is True
        before = len(env._env.events)
        for _ in range(4):
            absorbed = env.step("success")
            assert absorbed[0] == terminal[0]
            assert absorbed[1] == 0.0
            assert absorbed[2] is True
            assert absorbed[4]["is_success"] is True
        assert len(env._env.events) == before
    assert ids == list(range(50))
    assert env.evaluation_task_name == "move_object"


def test_reset_orders_reset_before_initial_state_and_settling():
    env = environment(episode_ids=[7])
    env._env.events.clear()
    env.reset(seed=9)
    assert env._env.events == [
        ("gym_reset", 9),
        ("seed", 9),
        ("reset",),
        ("set_init_state", 7),
        ("step", "dummy"),
        ("step", "dummy"),
    ]


def test_reset_clears_terminal_and_cycles_explicit_order():
    env = environment(episode_ids=[7, 2])
    env.reset()
    env.step("success")
    env.reset()
    assert env.evaluation_episode_id == 2
    before = len(env._env.events)
    assert env.step("normal")[2] is False
    assert len(env._env.events) == before + 1
    env.reset()
    assert env.evaluation_episode_id == 7


def test_no_episode_list_preserves_fixed_initial_state():
    env = environment(episode_index=13)
    for _ in range(2):
        obs, _ = env.reset()
        assert obs == {"formatted": 13}
        assert env.evaluation_episode_id == 13


def test_zero_settling_still_formats_selected_initial_state():
    env = environment(episode_ids=[8], num_steps_wait=0)
    assert env.reset()[0] == {"formatted": 8}


def test_failed_step_restores_reset_guard():
    env = environment(episode_ids=[0, 1])
    env.reset()
    with pytest.raises(RuntimeError, match="step failed"):
        env.step("raise")
    assert not env._in_step
    env.reset()
    assert env.evaluation_episode_id == 1


@pytest.mark.parametrize("seed", [True, -1, 1.5, [1, 2]])
def test_invalid_or_ambiguous_seed_does_not_consume_initial_state(seed):
    env = environment(episode_ids=[3, 4])
    with pytest.raises((TypeError, ValueError)):
        env.reset(seed=seed)
    env.reset(seed=5)
    assert env.evaluation_episode_id == 3


def test_single_seed_sequence_is_unwrapped_for_gym():
    env = environment(episode_ids=[0])
    env.reset(seed=[12])
    assert env._env.events[1:3] == [("gym_reset", 12), ("seed", 12)]


@pytest.mark.parametrize("ids", [[], [-1], [50], [1.5], [True]])
def test_invalid_initial_state_ids_fail(ids):
    with pytest.raises((TypeError, ValueError)):
        environment(episode_ids=ids)
