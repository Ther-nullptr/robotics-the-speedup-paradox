"""Bounded episode video capture with a caller-supplied synchronous writer.

Frames and summaries stay paired by task and initial-state ID. This module uses
only the standard library; the evaluator supplies rendering and video encoding.
"""

from copy import deepcopy
import math
from numbers import Integral, Real
from pathlib import Path


class EpisodeVideoRecorder:
    def __init__(self, output_dir, limit, fps, writer):
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 0:
            raise ValueError("video limit must be a nonnegative integer")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, Real)
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("video fps must be a finite positive number")
        if not callable(writer):
            raise TypeError("video writer must be callable")
        self.output_dir = Path(output_dir)
        self.limit = int(limit)
        self.fps = fps
        self.writer = writer
        self.paths = {}
        self.records = []
        self._pending = None

    def start(self, batch, rollout_index):
        """Select valid slots before rendering; return no callback if quota is full."""
        if self._pending is not None:
            raise RuntimeError(
                "finish the previous video rollout before starting another"
            )
        if (
            isinstance(rollout_index, bool)
            or not isinstance(rollout_index, Integral)
            or rollout_index < 0
        ):
            raise ValueError("rollout_index must be a nonnegative integer")
        pending = []
        reserved = {}
        for slot, (task, _, episode_ids) in enumerate(batch):
            if rollout_index >= len(episode_ids):
                continue
            if len(self.paths.get(task, [])) + reserved.get(task, 0) >= self.limit:
                continue
            if (
                not isinstance(task, str)
                or not task
                or Path(task).name != task
                or task in {".", ".."}
            ):
                raise ValueError("video task must be a single directory name")
            episode = episode_ids[rollout_index]
            if (
                isinstance(episode, bool)
                or not isinstance(episode, Integral)
                or episode < 0
            ):
                raise ValueError("video episode ID must be a nonnegative integer")
            pending.append(
                {
                    "slot": slot,
                    "task": task,
                    "init_state_id": int(episode),
                    "frames": [],
                }
            )
            reserved[task] = reserved.get(task, 0) + 1
        self._pending = pending
        if not pending:
            return None

        def capture(env):
            if self._pending is not pending:
                raise RuntimeError("video callback belongs to a finished rollout")
            rendered = env.render()
            for entry in pending:
                entry["frames"].append(deepcopy(rendered[entry["slot"]]))

        return capture

    def finish(self, rows):
        """Write selected episodes through their first terminal frame, then count them."""
        if self._pending is None:
            raise RuntimeError("start a video rollout before finishing it")
        by_episode = {}
        for row in rows:
            key = row["task"], row["init_state_id"]
            if key in by_episode:
                raise RuntimeError(f"duplicate video episode result: {key}")
            by_episode[key] = row
        for entry in self._pending:
            key = entry["task"], entry["init_state_id"]
            if key not in by_episode:
                raise RuntimeError(f"missing video episode result: {key}")
            row = by_episode[key]
            steps = row["primitive_steps"]
            if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
                raise ValueError("video primitive_steps must be a positive integer")
            count = (
                int(steps) + 1
            )  # Initial observation plus every step, including terminal.
            if len(entry["frames"]) < count:
                raise RuntimeError(f"video is missing its terminal frame: {key}")
            success = row["success"]
            if not isinstance(success, bool):
                raise ValueError("video success must be boolean")
            status = "success" if success else "fail"
            path = (
                self.output_dir / "videos" / key[0] / f"init_{key[1]:03d}_{status}.mp4"
            )
            if path.exists():
                raise FileExistsError(f"video already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writer(path, entry["frames"][:count], self.fps)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(
                    f"video writer did not produce a nonempty file: {path}"
                )
            self.paths.setdefault(key[0], []).append(str(path))
            self.records.append(
                {
                    "task": key[0],
                    "init_state_id": key[1],
                    "frame_count": count,
                    "fps": self.fps,
                    "path": str(path),
                    "success": success,
                }
            )
        self._pending = None
