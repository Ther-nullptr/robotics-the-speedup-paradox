"""Native LIBERO adapter contracts using CPU-only, in-memory environments."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/robotics_bench/simulators/libero.py"


def api():
    assert SOURCE.is_file(), "native LIBERO adapter is not implemented"
    spec = importlib.util.spec_from_file_location("tested_native_libero", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend(monkeypatch, tmp_path):
    events = []
    loads = []
    environments = []
    tasks = [
        SimpleNamespace(
            name=f"task_{index}",
            language=f"Move object {index} onto the tray.",
            problem_folder="libero_spatial",
            bddl_file=f"task_{index}.bddl",
            init_states_file=f"task_{index}.pruned_init",
        )
        for index in range(2)
    ]
    states = np.arange(12, dtype=float).reshape(3, 4)

    class Benchmark:
        n_tasks = 2

        def get_task(self, index):
            return tasks[index]

        def get_task_init_states(self, index):
            raise AssertionError("use explicit CPU torch.load instead")

    class Environment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.env = SimpleNamespace(control_timestep=0.05)
            self.actions = []
            self.success = False
            self.done = False
            self.tick = 0
            self.fail_step = False
            self.observation = {
                "agentview_image": np.zeros((2, 3, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((2, 3, 3), dtype=np.uint8),
                "robot0_gripper_qpos": np.array([1.0, 2.0]),
                "robot0_eef_pos": np.array([3.0, 4.0, 5.0]),
                "robot0_eef_quat": np.array([6.0, 7.0, 8.0, 9.0]),
            }
            environments.append(self)

        def _observation(self):
            self.observation["agentview_image"][0] = self.tick
            self.observation["agentview_image"][1] = self.tick + 1
            self.observation["robot0_eye_in_hand_image"][0] = self.tick + 20
            self.observation["robot0_eye_in_hand_image"][1] = self.tick + 21
            return self.observation

        def seed(self, seed):
            events.append(("seed", seed))

        def reset(self):
            events.append(("reset",))
            self.tick = 0
            self.success = False
            self.done = False
            return self._observation()

        def set_init_state(self, state):
            events.append(("set_init_state", list(state)))
            self.tick = 1
            return self._observation()

        def step(self, action):
            events.append(("step", list(action)))
            self.actions.append(list(action))
            self.tick += 1
            if self.fail_step:
                raise RuntimeError("native step failed")
            return self._observation(), 0.0, self.done, {}

        def check_success(self):
            return self.success

        def close(self):
            events.append(("close",))

    def load(path, **kwargs):
        loads.append((Path(path), kwargs))
        return states.copy()

    modules = {
        name: ModuleType(name)
        for name in (
            "libero",
            "libero.libero",
            "libero.libero.benchmark",
            "libero.libero.envs",
            "torch",
        )
    }
    modules["libero.libero"].get_libero_path = lambda name: str(tmp_path / name)
    modules["libero.libero.benchmark"].get_benchmark_dict = lambda: {
        "libero_spatial": Benchmark
    }
    modules["libero.libero.envs"].OffScreenRenderEnv = Environment
    modules["torch"].load = load
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        np=np,
        events=events,
        loads=loads,
        environments=environments,
        states=states,
        root=tmp_path,
        tasks=tasks,
    )


def test_import_needs_only_standard_library():
    assert SOURCE.is_file(), "native LIBERO adapter is not implemented"
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


def test_metadata_reads_trusted_initial_states_explicitly_on_cpu(backend):
    suite = api().LiberoSuite("libero_spatial")
    assert suite.tasks() == [
        {
            "id": index,
            "name": f"task_{index}",
            "description": f"Move object {index} onto the tray.",
            "initial_states": 3,
        }
        for index in range(2)
    ]
    assert backend.environments == []
    assert backend.loads == [
        (
            backend.root
            / "init_states"
            / "libero_spatial"
            / f"task_{index}.pruned_init",
            {"map_location": "cpu", "weights_only": False},
        )
        for index in range(2)
    ]


def test_reset_restores_state_then_settles_with_source_dummy_action(backend):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(1, env_seed=7)
    observation = simulator.reset(2)
    assert backend.events[:3] == [
        ("seed", 7),
        ("reset",),
        ("set_init_state", backend.states[2].tolist()),
    ]
    assert backend.events[3:] == [("step", [0, 0, 0, 0, 0, 0, -1])] * 10
    assert observation["primary_image"][0, 0, 0] == 11
    assert simulator.task_name == "task_1"
    assert simulator.description == "Move object 1 onto the tray."
    assert simulator.control_dt == 0.05
    assert backend.environments[0].kwargs == {
        "bddl_file_name": str(backend.root / "bddl_files/libero_spatial/task_1.bddl"),
        "camera_heights": 256,
        "camera_widths": 256,
    }


def test_reset_seed_override_and_zero_settling(backend):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0, settle_steps=0)
    observation = simulator.reset(0, seed=42)
    assert backend.events == [
        ("seed", 42),
        ("reset",),
        ("set_init_state", backend.states[0].tolist()),
    ]
    assert observation["primary_image"][0, 0, 0] == 1
    simulator.reset(1)
    assert backend.events[3] == ("seed", 0)


def test_policy_observation_preserves_raw_images_and_nine_dimensional_state(backend):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0, settle_steps=0)
    observation = simulator.reset(0)
    assert set(observation) == {"primary_image", "wrist_image", "proprio"}
    assert observation["primary_image"].dtype == backend.np.uint8
    assert observation["wrist_image"].dtype == backend.np.uint8
    assert observation["primary_image"][:, 0, 0].tolist() == [1, 2]
    assert observation["wrist_image"][:, 0, 0].tolist() == [21, 22]
    assert observation["proprio"].shape == (9,)
    assert observation["proprio"].tolist() == list(range(1, 10))
    simulator.step([0] * 7)
    assert observation["primary_image"][:, 0, 0].tolist() == [1, 2]


def test_proprio_preserves_native_float64_precision(backend):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0, settle_steps=0)
    native = backend.environments[0].observation
    expected = np.arange(1, 10, dtype=np.float64) + 2**-40
    native["robot0_gripper_qpos"][:] = expected[:2]
    native["robot0_eef_pos"][:] = expected[2:5]
    native["robot0_eef_quat"][:] = expected[5:]
    observation = simulator.reset(0)
    assert observation["proprio"].dtype == np.float64
    np.testing.assert_array_equal(observation["proprio"], expected)


def test_render_is_an_upright_cached_copy_without_stepping(backend):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0, settle_steps=0)
    observation = simulator.reset(0)
    observation["primary_image"][:] = 200
    before = list(backend.events)
    frame = simulator.render()
    assert frame[:, 0, 0].tolist() == [2, 1]
    frame[:] = 100
    assert simulator.render()[:, 0, 0].tolist() == [2, 1]
    assert backend.events == before


@pytest.mark.parametrize("success,done", [(True, True), (True, False), (False, True)])
def test_terminal_step_never_autoresets_or_steps_twice(backend, success, done):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0, settle_steps=0)
    simulator.reset(0)
    environment = backend.environments[0]
    environment.success, environment.done = success, done
    observation, actual_success, terminal = simulator.step([0.1] * 7)
    assert actual_success is success
    assert terminal is True
    assert observation["primary_image"][0, 0, 0] == 2
    assert simulator.render()[0, 0, 0] == 3
    assert environment.actions == [[0.1] * 7]
    with pytest.raises(RuntimeError, match="terminal|terminated"):
        simulator.step([0] * 7)
    assert environment.actions == [[0.1] * 7]
    assert [event for event in backend.events if event[0] == "reset"] == [("reset",)]
    simulator.reset(1)
    assert simulator.step([0] * 7)[1:] == (False, False)


@pytest.mark.parametrize(
    "action", [[0] * 6, [[0] * 7], [float("nan")] * 7, [float("inf")] * 7]
)
def test_invalid_actions_are_rejected_before_physical_step(backend, action):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0, settle_steps=0)
    simulator.reset(0)
    with pytest.raises(ValueError, match="action"):
        simulator.step(action)
    assert backend.environments[0].actions == []


@pytest.mark.parametrize("task_id", [-1, 2, True, "0"])
def test_invalid_task_ids_do_not_create_an_environment(backend, task_id):
    with pytest.raises(ValueError, match="task_id"):
        api().LiberoSuite("libero_spatial").make_simulator(task_id)
    assert backend.environments == []


@pytest.mark.parametrize(
    "kwargs",
    [{"env_seed": -1}, {"env_seed": True}, {"settle_steps": -1}, {"settle_steps": 1.5}],
)
def test_invalid_reset_configuration_does_not_create_an_environment(backend, kwargs):
    with pytest.raises(ValueError):
        api().LiberoSuite("libero_spatial").make_simulator(0, **kwargs)
    assert backend.environments == []


@pytest.mark.parametrize("state_id", [-1, 3, True, "0"])
def test_invalid_initial_state_does_not_reset_or_step(backend, state_id):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0)
    with pytest.raises(ValueError, match="init_state_id"):
        simulator.reset(state_id)
    assert backend.events == []


def test_unknown_suite_has_actionable_error(backend):
    with pytest.raises(ValueError, match="suite"):
        api().LiberoSuite("unknown")


def test_close_is_idempotent_and_disallows_further_use(backend):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0, settle_steps=0)
    with pytest.raises(RuntimeError, match="reset"):
        simulator.render()
    with pytest.raises(RuntimeError, match="reset"):
        simulator.step([0] * 7)
    simulator.reset(0)
    simulator.close()
    simulator.close()
    assert backend.events.count(("close",)) == 1
    with pytest.raises(RuntimeError, match="closed"):
        simulator.reset(0)
    with pytest.raises(RuntimeError, match="closed"):
        simulator.step([0] * 7)
    with pytest.raises(RuntimeError, match="closed"):
        simulator.render()


def test_native_errors_propagate_without_retrying_actions(backend):
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0, settle_steps=0)
    simulator.reset(0)
    backend.environments[0].fail_step = True
    with pytest.raises(RuntimeError, match="native step failed"):
        simulator.step([0] * 7)
    assert backend.environments[0].actions == [[0] * 7]


def test_runner_budget_excludes_settling_and_uses_language_instruction(backend):
    from robotics_bench.protocols.static_runner import run_episode

    requests = []
    engine = SimpleNamespace(
        reset=lambda episode: None,
        infer_chunk=lambda observation, task, seed: requests.append(task) or [[0] * 7],
    )
    simulator = api().LiberoSuite("libero_spatial").make_simulator(0)
    result = run_episode(
        engine,
        simulator,
        task="task_0",
        init_state_id=0,
        env_seed=0,
        sampling_seed=0,
        max_steps=2,
        n_action_steps=1,
    )
    assert result["primitive_steps"] == 2
    assert result["inference_calls"] == 2
    assert result["termination_reason"] == "budget_exhausted"
    assert len(backend.environments[0].actions) == 12
    assert requests == ["Move object 0 onto the tray."] * 2
