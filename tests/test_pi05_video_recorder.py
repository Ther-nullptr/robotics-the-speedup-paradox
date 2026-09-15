"""CPU-only episode video selection, frame ownership, and write accounting."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

MODULE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/static/pi05_libero/video_recorder.py"
)


def api():
    assert MODULE.is_file(), "video_recorder.py is not implemented"
    spec = importlib.util.spec_from_file_location("video_recorder_tested", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEnv:
    def __init__(self, frames):
        self.frames = frames
        self.calls = 0

    def render(self):
        self.calls += 1
        return self.frames


class FakeWriter:
    def __init__(self):
        self.calls = []

    def __call__(self, path, frames, fps):
        self.calls.append((Path(path), frames, fps))
        Path(path).write_bytes(b"fake video bytes")


def row(task, initial, steps=1, success=True):
    return {
        "task": task,
        "init_state_id": initial,
        "primitive_steps": steps,
        "success": success,
    }


def recorder(tmp_path, limit=1, writer=None):
    writer = writer or FakeWriter()
    return api().EpisodeVideoRecorder(tmp_path, limit, 30, writer), writer


def test_import_has_no_image_or_model_dependencies():
    assert MODULE.is_file(), "video_recorder.py is not implemented"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import runpy,sys;runpy.run_path(sys.argv[1]);"
            "assert not any(k in sys.modules for k in ('torch','numpy','imageio'))",
            str(MODULE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_zero_quota_never_returns_callback_or_creates_directory(tmp_path):
    output = tmp_path / "output"
    rec, writer = recorder(output, limit=0)
    assert rec.start([("task", "language", [0])], 0) is None
    rec.finish([row("task", 0)])
    assert writer.calls == []
    assert rec.paths == {}
    assert rec.records == []
    assert not output.exists()


def test_frames_are_owned_and_include_terminal_frame_but_not_padding(tmp_path):
    rec, writer = recorder(tmp_path)
    callback = rec.start([("task", "language", [7])], 0)
    frame = [0]
    env = FakeEnv([frame])
    for value in range(5):
        frame[0] = value
        callback(env)
    rec.finish([row("task", 7, steps=2)])
    path, frames, fps = writer.calls[0]
    assert frames == [[0], [1], [2]]
    assert fps == 30
    assert path == tmp_path / "videos/task/init_007_success.mp4"
    assert rec.paths == {"task": [str(path)]}
    assert rec.records == [
        {
            "task": "task",
            "init_state_id": 7,
            "frame_count": 3,
            "fps": 30,
            "path": str(path),
            "success": True,
        }
    ]


def test_same_task_slots_share_quota_and_unselected_frames_are_not_copied(tmp_path):
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("unselected frame was copied")

    rec, writer = recorder(tmp_path, limit=1)
    callback = rec.start([("task", "a", [0]), ("task", "b", [1])], 0)
    env = FakeEnv([[4], CannotCopy()])
    callback(env)
    callback(env)
    rec.finish([row("task", 1), row("task", 0)])
    assert len(writer.calls) == 1
    assert rec.records[0]["init_state_id"] == 0


def test_quota_is_per_task_across_batches_and_finished_slots_are_ignored(tmp_path):
    rec, writer = recorder(tmp_path, limit=2)
    batch = [("task", "a", [0]), ("other", "b", [3, 4])]
    callback = rec.start(batch, 0)
    callback(FakeEnv([[0], [3]]))
    callback(FakeEnv([[1], [4]]))
    rec.finish([row("task", 0), row("other", 3)])
    callback = rec.start(batch, 1)
    callback(FakeEnv([[99], [4]]))
    callback(FakeEnv([[99], [5]]))
    rec.finish([row("other", 4)])
    callback = rec.start([("task", "c", [8]), ("other", "d", [9])], 0)
    callback(FakeEnv([[8], [9]]))
    callback(FakeEnv([[9], [10]]))
    rec.finish([row("task", 8), row("other", 9)])
    assert [item["init_state_id"] for item in rec.records] == [0, 3, 4, 8]
    assert rec.start([("task", "e", [12]), ("other", "f", [13])], 0) is None
    rec.finish([row("task", 12), row("other", 13)])
    assert len(writer.calls) == 4


def test_failed_episode_is_saved_with_fail_suffix(tmp_path):
    rec, writer = recorder(tmp_path)
    callback = rec.start([("task", "a", [2])], 0)
    callback(FakeEnv([[0]]))
    callback(FakeEnv([[1]]))
    rec.finish([row("task", 2, success=False)])
    assert writer.calls[0][0].name == "init_002_fail.mp4"
    assert rec.records[0]["success"] is False


@pytest.mark.parametrize("behavior", ["raise", "missing", "empty"])
def test_failed_writer_does_not_count_or_advertise_video(tmp_path, behavior):
    def writer(path, frames, fps):
        if behavior == "raise":
            raise RuntimeError("encoder failed")
        if behavior == "empty":
            Path(path).touch()

    rec, _ = recorder(tmp_path, writer=writer)
    callback = rec.start([("task", "a", [0])], 0)
    callback(FakeEnv([[0]]))
    callback(FakeEnv([[1]]))
    with pytest.raises((RuntimeError, OSError), match="encoder|video"):
        rec.finish([row("task", 0)])
    assert rec.paths == {}
    assert rec.records == []


def test_existing_video_is_not_overwritten(tmp_path):
    rec, writer = recorder(tmp_path)
    target = tmp_path / "videos/task/init_000_success.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"keep")
    callback = rec.start([("task", "a", [0])], 0)
    callback(FakeEnv([[0]]))
    callback(FakeEnv([[1]]))
    with pytest.raises((FileExistsError, RuntimeError)):
        rec.finish([row("task", 0)])
    assert target.read_bytes() == b"keep"
    assert writer.calls == []
    assert rec.records == []


def test_missing_terminal_frame_is_rejected(tmp_path):
    rec, writer = recorder(tmp_path)
    callback = rec.start([("task", "a", [0])], 0)
    callback(FakeEnv([[0]]))
    with pytest.raises(RuntimeError, match="frame"):
        rec.finish([row("task", 0, steps=1)])
    assert writer.calls == []


def test_missing_or_duplicate_episode_identity_is_rejected(tmp_path):
    rec, writer = recorder(tmp_path)
    callback = rec.start([("task", "a", [0])], 0)
    callback(FakeEnv([[0]]))
    callback(FakeEnv([[1]]))
    with pytest.raises(RuntimeError, match="episode"):
        rec.finish([row("task", 1)])
    assert writer.calls == []


@pytest.mark.parametrize("limit,fps", [(-1, 30), (True, 30), (1, 0), (1, float("nan"))])
def test_invalid_recording_options_fail_early(tmp_path, limit, fps):
    with pytest.raises((TypeError, ValueError)):
        api().EpisodeVideoRecorder(tmp_path, limit, fps, FakeWriter())
