"""Native single-environment RoboCasa adapter for the Cosmos Policy case."""

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import random
import sys


TASK_MAX_STEPS = {
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}
CAMERAS = {
    "primary_image": "robot0_agentview_left_image",
    "secondary_image": "robot0_agentview_right_image",
    "wrist_image": "robot0_eye_in_hand_image",
}


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def import_backend(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    package = importlib.import_module("robocasa")
    if not Path(package.__file__).resolve().is_relative_to(source / "robocasa"):
        raise RuntimeError("RoboCasa was imported outside the requested source")
    robosuite = importlib.import_module("robosuite")
    if robosuite.__version__ != "1.5.1":
        raise RuntimeError("This RoboCasa controller adapter requires robosuite 1.5.1")
    return robosuite


def controller_action(env, action):
    """Map native 7D manipulation to the verified mobile controller layout."""
    np = importlib.import_module("numpy")
    action = np.asarray(action, dtype=float)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("action must contain exactly seven finite numbers")
    if env.action_dim != 12 or len(env.robots) != 1:
        raise ValueError("Unsupported RoboCasa controller action dimension")
    robot = env.robots[0]
    split = robot._action_split_indexes
    if (
        robot.composite_controller.name != "HYBRID_MOBILE_BASE"
        or split.get("right") != (0, 6)
        or split.get("right_gripper") != (6, 7)
        or set(split) != {"right", "right_gripper", "base", "torso"}
        or sorted(i for key in ("base", "torso") for i in range(*split[key]))
        != list(range(7, 11))
    ):
        raise ValueError("Unsupported RoboCasa controller layout")
    return np.concatenate([action, np.array([0, 0, 0, 0, -1], dtype=float)])


class NativeRoboCasaSimulator:
    """Create a fresh seeded scene per episode and read language after reset.

    The constructor does not instantiate an environment. Each reset creates one
    native environment and calls its public reset exactly once, following the
    reference evaluator. Ten settling steps are excluded from control counts.
    """

    def __init__(
        self,
        source,
        controller_config,
        task_name,
        *,
        layout_id=1,
        style_id=1,
        settle_steps=10,
        image_size=224,
    ):
        if task_name not in TASK_MAX_STEPS:
            raise ValueError("Task is outside the Cosmos RoboCasa 24-task case")
        self.source = Path(source).expanduser().resolve()
        self.controller_config = Path(controller_config).expanduser().resolve()
        self.task_name = task_name
        self.layout_id = _integer(layout_id, "layout_id")
        self.style_id = _integer(style_id, "style_id")
        self.settle_steps = _integer(settle_steps, "settle_steps")
        self.image_size = image_size
        self.description = None
        self.control_dt = None
        self.episode_metadata = None
        self._env = None
        self._frame = None
        self._ready = False
        self._closed = False
        self._initial_state = None
        self._initial_xml = None

    def _observation(self, native):
        np = importlib.import_module("numpy")
        result = {
            key: np.array(native[name], copy=True) for key, name in CAMERAS.items()
        }
        result["proprio"] = np.concatenate(
            [
                native["robot0_gripper_qpos"],
                native["robot0_eef_pos"],
                native["robot0_eef_quat"],
            ]
        )
        self._frame = np.concatenate(
            [result[key][::-1] for key in CAMERAS], axis=1
        ).copy()
        return result

    def reset(self, init_state_id, seed=None):
        if self._closed:
            raise RuntimeError("RoboCasa simulator is closed")
        _integer(init_state_id, "init_state_id")
        _integer(seed, "env_seed")
        self._ready = False
        self._frame = None
        self.description = None
        self.episode_metadata = None
        self._initial_state = None
        self._initial_xml = None
        if self._env is not None:
            self._env.close()
            self._env = None
        np = importlib.import_module("numpy")
        backend = import_backend(self.source)
        # No aliases: a new upstream task is not necessarily the trained task.
        if self.task_name not in backend.ALL_ENVIRONMENTS:
            raise ValueError(
                f"Task {self.task_name} is absent from this RoboCasa source"
            )
        random.seed(seed)
        np.random.seed(seed)
        with self.controller_config.open("rb") as stream:
            controller = pickle.load(stream)
        if not isinstance(controller, dict) or controller.get("type") != "OSC_POSE":
            raise ValueError("Expected the native Cosmos OSC_POSE controller config")
        self._env = backend.make(
            env_name=self.task_name,
            robots="PandaMobile",
            controller_configs=deepcopy(controller),
            camera_names=[name.removesuffix("_image") for name in CAMERAS.values()],
            camera_widths=self.image_size,
            camera_heights=self.image_size,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
            seed=seed,
            obj_instance_split="B",
            generative_textures=None,
            randomize_cameras=False,
            layout_and_style_ids=((self.layout_id, self.style_id),),
            translucent_robot=False,
        )
        try:
            native = self._env.reset()
            meta = self._env.get_ep_meta()
            self.description = meta["lang"]
            controller_action(self._env, np.zeros(7))
            for _ in range(self.settle_steps):
                native, _, done, _ = self._env.step(np.zeros(self._env.action_dim))
                if done:
                    raise RuntimeError("RoboCasa terminated during reset settling")
            if self._env._check_success():
                raise RuntimeError(
                    "RoboCasa task is already successful before policy control"
                )
            observation = self._observation(native)
            self.control_dt = float(self._env.control_timestep)
            self._initial_state = np.array(
                self._env.sim.get_state().flatten(), copy=True
            )
            self._initial_xml = self._env.model.get_xml()
            digest = hashlib.sha256()
            for key, value in observation.items():
                digest.update(key.encode())
                digest.update(str(value.dtype).encode())
                digest.update(value.tobytes())
            self.episode_metadata = {
                "task": self.task_name,
                "init_state_id": init_state_id,
                "env_seed": seed,
                "layout_id": self.layout_id,
                "style_id": self.style_id,
                "description": self.description,
                "native_episode": meta,
                "initial_state_sha256": hashlib.sha256(
                    self._initial_state.tobytes()
                ).hexdigest(),
                "initial_xml_sha256": hashlib.sha256(
                    self._initial_xml.encode()
                ).hexdigest(),
                "initial_observation_sha256": digest.hexdigest(),
                "controller_layout": dict(self._env.robots[0]._action_split_indexes),
                "control_dt_seconds": self.control_dt,
                "settle_steps": self.settle_steps,
            }
            self._ready = True
            return observation
        except BaseException:
            self._env.close()
            self._env = None
            raise

    def save_initialization(self, directory):
        """Save physics state, scene XML and native metadata for paired-run auditing."""
        if not self._ready:
            raise RuntimeError("Reset the environment before saving initialization")
        np = importlib.import_module("numpy")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.save(directory / "state.npy", self._initial_state, allow_pickle=False)
        (directory / "model.xml").write_text(self._initial_xml)
        (directory / "episode.json").write_text(
            json.dumps(
                self.episode_metadata,
                indent=2,
                allow_nan=False,
                default=lambda value: (
                    value.tolist() if hasattr(value, "tolist") else str(value)
                ),
            )
            + "\n"
        )

    def step(self, action):
        if not self._ready or self._closed:
            raise RuntimeError("Reset the RoboCasa simulator before stepping")
        native_action = controller_action(self._env, action)
        self._ready = False
        native, _, done, _ = self._env.step(native_action)
        success = bool(self._env._check_success())
        observation = self._observation(native)
        terminated = bool(done or success)
        self._ready = not terminated
        return observation, success, terminated

    def render(self):
        if self._closed or self._frame is None:
            raise RuntimeError("Reset the RoboCasa simulator before rendering")
        return self._frame.copy()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
        self._ready = False
        self._closed = True
        self._frame = None
