"""Read-only RoboTwin source/asset adapter for the LingBot static case."""

from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys


def absolute_eef_action(relative, initial):
    import numpy as np
    from scipy.spatial.transform import Rotation

    relative, initial = np.asarray(relative), np.asarray(initial)
    if (
        relative.shape != (16,)
        or initial.shape != (16,)
        or not np.isfinite(relative).all()
    ):
        raise ValueError("Expected sixteen finite EEF action values")
    result = []
    for start in (0, 8):
        delta, origin = relative[start : start + 8], initial[start : start + 8]
        if min(np.linalg.norm(delta[3:7]), np.linalg.norm(origin[3:7])) < 1e-12:
            raise ValueError("Invalid zero quaternion in EEF action")
        rotation = (
            Rotation.from_quat(origin[3:7]) * Rotation.from_quat(delta[3:7])
        ).as_quat()
        result.extend([*(origin[:3] + delta[:3]), *rotation, delta[7]])
    return np.asarray(result, dtype=np.float64)


class RoboTwinSimulator:
    """One task instance; native expert filtering selects eligible scene seeds."""

    def __init__(
        self,
        source,
        task,
        *,
        task_config="demo_clean",
        output_dir,
        max_initialization_attempts=32,
    ):
        if not task.isidentifier():
            raise ValueError("task must be a Python identifier")
        if task_config not in {"demo_clean", "demo_randomized"}:
            raise ValueError("Unsupported RoboTwin task configuration")
        self.source = Path(source).expanduser().resolve()
        self.output = Path(output_dir).resolve()
        self.task_name = task
        self.task_config = task_config
        self.max_attempts = max_initialization_attempts
        self.description = None
        self.step_count = 0
        self._env = None
        self._frame = None
        self._initial_pose = None
        self._original_cwd = Path.cwd()
        self._original_path = sys.path[:]
        self.metadata = None

    def _configuration(self):
        import yaml

        def read(path):
            return yaml.safe_load(path.read_text())

        config = read(self.source / "task_config" / (self.task_config + ".yml"))
        if config.get("embodiment") != ["aloha-agilex"]:
            raise ValueError(
                "This checkpoint case requires the aloha-agilex embodiment"
            )
        embodiments = read(self.source / "task_config/_embodiment_config.yml")
        robot = (self.source / embodiments["aloha-agilex"]["file_path"]).resolve()
        robot_config = read(robot / "config.yml")
        cameras = read(self.source / "task_config/_camera_config.yml")
        camera = cameras[config["camera"]["head_camera_type"]]
        config.update(
            task_name=self.task_name,
            task_config=self.task_config,
            policy_name="lingbot_va",
            eval_mode=True,
            render_freq=0,
            save_path=str(self.output / "native"),
            save_data=False,
            collect_data=False,
            eval_video_log=False,
            eval_video_save_dir=None,
            left_robot_file=str(robot),
            right_robot_file=str(robot),
            dual_arm_embodied=True,
            left_embodiment_config=deepcopy(robot_config),
            right_embodiment_config=deepcopy(robot_config),
            head_camera_h=camera["h"],
            head_camera_w=camera["w"],
        )
        config["data_type"]["pointcloud"] = False
        return config

    def _observation(self):
        import numpy as np

        native = self._env.get_obs()
        values = {
            "observation.images.cam_high": native["observation"]["head_camera"]["rgb"],
            "observation.images.cam_left_wrist": native["observation"]["left_camera"][
                "rgb"
            ],
            "observation.images.cam_right_wrist": native["observation"]["right_camera"][
                "rgb"
            ],
        }
        result = {key: np.array(value, copy=True) for key, value in values.items()}
        result["observation.state"] = np.array(
            native["joint_action"]["vector"], copy=True
        )
        result["task"] = self.description
        # Record each policy command, independently of the model's keyframe sampling.
        self._frame = np.concatenate([result[key] for key in values], axis=1).copy()
        return native, result

    def prepare(self, seed, episode_index=0):
        import numpy as np

        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self._env is not None:
            self._env.close_env()
            self._env = None
        self._frame = None
        self._initial_pose = None
        self.description = None
        if str(self.source) not in sys.path:
            sys.path.insert(0, str(self.source))
        os.chdir(self.source)
        module = importlib.import_module("envs." + self.task_name)
        if not Path(module.__file__).resolve().is_relative_to(self.source / "envs"):
            raise RuntimeError(
                "RoboTwin was imported from a different source directory"
            )
        native_type = getattr(module, self.task_name)
        unstable_error = importlib.import_module(
            "envs.utils.create_actor"
        ).UnStableError
        instructions = importlib.import_module(
            "description.utils.generate_episode_instructions"
        )
        config = self._configuration()
        attempts = []
        for candidate in range(seed, seed + self.max_attempts):
            self._env = native_type()
            random.seed(candidate)
            try:
                self._env.setup_demo(
                    now_ep_num=episode_index, seed=candidate, is_test=True, **config
                )
                episode_info = self._env.play_once()
                eligible = bool(self._env.plan_success and self._env.check_success())
                attempts.append({"seed": candidate, "eligible": eligible})
            except unstable_error:
                attempts.append(
                    {"seed": candidate, "eligible": False, "reason": "unstable_scene"}
                )
                eligible = False
            finally:
                self._env.close_env()
                self._env = None
            if eligible:
                break
        else:
            raise RuntimeError(
                f"No eligible scene within {self.max_attempts} initialization attempts"
            )
        self._env = native_type()
        random.seed(candidate)
        self._env.setup_demo(
            now_ep_num=episode_index, seed=candidate, is_test=True, **config
        )
        descriptions = instructions.generate_episode_descriptions(
            self.task_name, [episode_info["info"]], 100
        )
        self.description = str(np.random.choice(descriptions[0]["seen"]))
        self._env.set_instruction(instruction=self.description)
        native, observation = self._observation()
        state = native["endpose"]
        self._initial_pose = np.asarray(
            list(state["left_endpose"])
            + [state["left_gripper"]]
            + list(state["right_endpose"])
            + [state["right_gripper"]],
            dtype=np.float64,
        )
        self.step_count = 0
        if self._env.take_action_cnt != 0:
            raise RuntimeError("Prepared RoboTwin episode has nonzero control count")
        digest = hashlib.sha256()
        for key, value in observation.items():
            if hasattr(value, "tobytes"):
                digest.update(key.encode())
                digest.update(value.tobytes())
        self.metadata = {
            "task": self.task_name,
            "task_config": self.task_config,
            "init_state_id": candidate,
            "requested_seed": seed,
            "accepted_seed": candidate,
            "initialization_attempts": attempts,
            "description": self.description,
            "initial_eef_pose": self._initial_pose.tolist(),
            "initial_observation_sha256": digest.hexdigest(),
            "native_step_limit": int(self._env.step_lim),
            "physics_timestep_seconds": float(self._env.scene.get_timestep()),
            "step_unit": "robotwin_take_action_commands",
        }
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / f"initialization-{episode_index:06d}.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n"
        )
        return observation

    def step(self, relative_action):
        if self._env is None or self._initial_pose is None:
            raise RuntimeError("Prepare RoboTwin before stepping")
        before = self._env.take_action_cnt
        action = absolute_eef_action(relative_action, self._initial_pose)
        self._env.take_action(action, action_type="ee")
        if self._env.take_action_cnt != before + 1:
            raise RuntimeError("RoboTwin did not accept exactly one control command")
        self.step_count += 1
        _, observation = self._observation()
        return observation, bool(self._env.eval_success)

    def render(self):
        if self._frame is None:
            raise RuntimeError("Prepare RoboTwin before rendering")
        return self._frame.copy()

    def close(self):
        try:
            if self._env is not None:
                self._env.close_env(clear_cache=True)
                self._env = None
        finally:
            os.chdir(self._original_cwd)
            sys.path[:] = self._original_path
