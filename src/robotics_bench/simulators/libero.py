"""Native LIBERO environments for the Cosmos single-environment static case.

Importing this module needs only the standard library. LIBERO, Torch, and NumPy
are loaded when the optional simulator or its task metadata is requested.
"""

from importlib import import_module
from numbers import Integral
from pathlib import Path


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


class LiberoSuite:
    """Expose native task ordering and trusted local initial-state assets."""

    def __init__(self, suite_name):
        if not isinstance(suite_name, str) or not suite_name.strip():
            raise ValueError("suite_name must be a nonempty LIBERO suite name")
        benchmark = import_module("libero.libero.benchmark")
        suites = benchmark.get_benchmark_dict()
        suite_name = suite_name.lower()
        if suite_name not in suites:
            raise ValueError(f"unknown LIBERO suite: {suite_name}")
        self._suite = suites[suite_name]()
        self._get_path = import_module("libero.libero").get_libero_path
        self._states = {}

    def _task(self, task_id):
        task_id = _integer(task_id, "task_id")
        if task_id >= self._suite.n_tasks:
            raise ValueError(f"task_id must be less than {self._suite.n_tasks}")
        return self._suite.get_task(task_id)

    def _initial_states(self, task_id):
        if task_id not in self._states:
            task = self._task(task_id)
            path = (
                Path(self._get_path("init_states"))
                / task.problem_folder
                / task.init_states_file
            )
            # These are the caller's trusted LIBERO assets. Explicit arguments
            # preserve compatibility across Torch's weights_only default change.
            torch = import_module("torch")
            self._states[task_id] = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
        return self._states[task_id]

    def tasks(self):
        """Return metadata without constructing a renderer or environment."""
        return [
            {
                "id": task_id,
                "name": task.name,
                "description": task.language,
                "initial_states": len(self._initial_states(task_id)),
            }
            for task_id in range(self._suite.n_tasks)
            for task in [self._task(task_id)]
        ]

    def make_simulator(self, task_id, *, env_seed=0, settle_steps=10):
        task = self._task(task_id)
        env_seed = _integer(env_seed, "env_seed")
        settle_steps = _integer(settle_steps, "settle_steps")
        states = self._initial_states(task_id)
        env_type = import_module("libero.libero.envs").OffScreenRenderEnv
        path = Path(self._get_path("bddl_files")) / task.problem_folder / task.bddl_file
        environment = env_type(
            bddl_file_name=str(path), camera_heights=256, camera_widths=256
        )
        try:
            return NativeLiberoSimulator(
                environment,
                task,
                states,
                env_seed=env_seed,
                settle_steps=settle_steps,
            )
        except BaseException:
            environment.close()
            raise


class NativeLiberoSimulator:
    """One native environment with explicit reset and no automatic reset.

    Policy observations preserve raw camera orientation and concatenate gripper
    qpos, end-effector position, then its xyzw quaternion. Only replay rendering
    flips the primary camera upright. Reset settling is outside rollout budgets.
    """

    def __init__(self, environment, task, initial_states, *, env_seed, settle_steps):
        self._env = environment
        self._states = initial_states
        self._env_seed = env_seed
        self._settle_steps = settle_steps
        self.task_name = task.name
        self.description = task.language
        self.control_dt = float(environment.env.control_timestep)
        self._closed = False
        self._ready = False
        self._terminated = False
        self._frame = None

    def _require_open(self):
        if self._closed:
            raise RuntimeError("LIBERO simulator is closed")

    def _observation(self, native):
        np = import_module("numpy")
        primary = np.array(native["agentview_image"], copy=True)
        wrist = np.array(native["robot0_eye_in_hand_image"], copy=True)
        proprio = np.concatenate(
            (
                native["robot0_gripper_qpos"],
                native["robot0_eef_pos"],
                native["robot0_eef_quat"],
            )
        )
        self._frame = primary[::-1].copy()
        return {"primary_image": primary, "wrist_image": wrist, "proprio": proprio}

    def reset(self, init_state_id, seed=None):
        self._require_open()
        init_state_id = _integer(init_state_id, "init_state_id")
        if init_state_id >= len(self._states):
            raise ValueError(f"init_state_id must be less than {len(self._states)}")
        seed = self._env_seed if seed is None else _integer(seed, "seed")
        self._ready = False
        self._terminated = False
        self._frame = None
        self._env.seed(seed)
        self._env.reset()
        native = self._env.set_init_state(self._states[init_state_id])
        for _ in range(self._settle_steps):
            native, _, done, _ = self._env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                self._terminated = True
                raise RuntimeError(
                    "LIBERO environment terminated during reset settling"
                )
        observation = self._observation(native)
        self._ready = True
        return observation

    def step(self, action):
        self._require_open()
        if self._terminated:
            raise RuntimeError("LIBERO episode is terminal; call reset before stepping")
        if not self._ready:
            raise RuntimeError("call reset before stepping the LIBERO simulator")
        np = import_module("numpy")
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("action must contain exactly seven finite numbers")
        # A native exception may follow a partial physical step. Require reset
        # rather than allow a retry to silently duplicate that action.
        self._ready = False
        native, _, done, _ = self._env.step(action)
        success = bool(self._env.check_success())
        self._terminated = bool(done or success)
        observation = self._observation(native)
        self._ready = True
        return observation, success, self._terminated

    def render(self):
        """Return a copy of the latest upright RGB frame without advancing time."""
        self._require_open()
        if self._frame is None:
            raise RuntimeError("call reset before rendering the LIBERO simulator")
        return self._frame.copy()

    def close(self):
        if not self._closed:
            self._closed = True
            self._ready = False
            self._frame = None
            self._env.close()
