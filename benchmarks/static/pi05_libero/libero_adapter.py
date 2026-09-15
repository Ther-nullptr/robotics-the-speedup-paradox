"""Explicit LIBERO episode resets with the native LeRobot observation/action path.

Optional simulator imports occur only in the public factory. The small subclass
suppresses LeRobot's step-internal autoreset, so vector padding cannot consume
initial states or advance the simulator after an episode has ended.
"""

from numbers import Integral


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _scalar_seed(seed):
    if seed is None:
        return None
    if not isinstance(seed, Integral) and not isinstance(seed, (str, bytes)):
        try:
            values = list(seed)
        except TypeError:
            values = None
        if values is not None:
            if len(values) != 1:
                raise ValueError(
                    "pass one scalar seed per environment, not a seed list"
                )
            seed = values[0]
    return _nonnegative_integer(seed, "seed")


def build_adapter_class(base_cls, gym_reset, dummy_action):
    """Build the adapter against an explicit base; fakes need no GPU packages."""

    class ExplicitResetLiberoEnv(base_cls):
        def __init__(self, *args, episode_ids=None, **kwargs):
            ids = (
                [kwargs.get("episode_index", 0)]
                if episode_ids is None
                else list(episode_ids)
            )
            if not ids:
                raise ValueError("episode_ids must not be empty")
            self._evaluation_episode_ids = [
                _nonnegative_integer(value, "episode id") for value in ids
            ]
            self._evaluation_episode_ptr = 0
            self.evaluation_episode_id = None
            self._in_step = False
            self._terminal_transition = None
            self._last_observation = None
            self._last_info = {}
            kwargs["episode_index"] = self._evaluation_episode_ids[0]
            # Current LeRobot constructor resets only its low-level environment.
            super().__init__(*args, **kwargs)
            self.evaluation_task_name = self.task
            if self._init_states is None or any(
                value >= len(self._init_states)
                for value in self._evaluation_episode_ids
            ):
                close = getattr(self, "close", None)
                if callable(close):
                    close()
                raise ValueError(
                    "episode_ids must index the available LIBERO initial states"
                )

        def reset(self, seed=None, **kwargs):
            if self._in_step:
                # LeRobot.step computes its terminal observation before calling
                # self.reset; that return value is ignored by the base method.
                return self._last_observation, dict(self._last_info)
            seed = _scalar_seed(seed)
            episode = self._evaluation_episode_ids[
                self._evaluation_episode_ptr % len(self._evaluation_episode_ids)
            ]
            gym_reset(self, seed=seed)
            self._env.seed(seed)
            self._env.reset()
            raw = self._env.set_init_state(self._init_states[episode])
            for _ in range(self.num_steps_wait):
                raw, _, _, _ = self._env.step(dummy_action())
            observation = self._format_raw_obs(raw)
            self._init_state_id = episode
            self.episode_index = episode
            self.evaluation_episode_id = episode
            self._evaluation_episode_ptr += 1
            self._terminal_transition = None
            self._last_observation = observation
            self._last_info = {
                "is_success": False,
                "evaluation_episode_id": episode,
                "evaluation_task_name": self.evaluation_task_name,
            }
            return observation, dict(self._last_info)

        def step(self, action):
            if self._terminal_transition is not None:
                observation, _, terminated, truncated, info = self._terminal_transition
                return observation, 0.0, terminated, truncated, dict(info)
            if self.evaluation_episode_id is None:
                raise RuntimeError("reset must be called before step")
            self._in_step = True
            try:
                observation, reward, terminated, truncated, info = super().step(action)
            finally:
                self._in_step = False
            info = dict(info)
            info.update(
                evaluation_episode_id=self.evaluation_episode_id,
                evaluation_task_name=self.evaluation_task_name,
            )
            self._last_observation = observation
            self._last_info = info
            result = observation, reward, terminated, truncated, info
            if terminated or truncated:
                self._terminal_transition = result
            return result

    return ExplicitResetLiberoEnv


def make_lerobot_libero_env(
    suite_name, task_name, init_state_id, gym_kwargs, episode_ids=None
):
    """Replacement for the compatible evaluator's LeRobot environment factory."""
    import gymnasium as gym
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if suite_name not in suites:
        raise ValueError(f"Unknown LIBERO suite: {suite_name}")
    suite = suites[suite_name]()
    task_ids = [
        index
        for index in range(suite.n_tasks)
        if suite.get_task(index).name == task_name
    ]
    if len(task_ids) != 1:
        raise ValueError(
            f"Expected exactly one task named {task_name!r} in {suite_name!r}"
        )
    adapter = build_adapter_class(LiberoEnv, gym.Env.reset, get_libero_dummy_action)
    options = {
        name: gym_kwargs[name]
        for name in (
            "camera_name",
            "camera_name_mapping",
            "observation_width",
            "observation_height",
            "visualization_width",
            "visualization_height",
            "num_steps_wait",
        )
        if name in gym_kwargs
    }
    env = adapter(
        task_suite=suite,
        task_id=task_ids[0],
        task_suite_name=suite_name,
        obs_type=str(gym_kwargs.get("obs_type", "pixels_agent_pos")),
        render_mode=str(gym_kwargs.get("render_mode", "rgb_array")),
        episode_index=init_state_id,
        episode_ids=episode_ids,
        **options,
    )
    if "max_episode_steps" in gym_kwargs:
        env._max_episode_steps = int(gym_kwargs["max_episode_steps"])
    return env
