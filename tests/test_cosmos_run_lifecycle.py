"""Exercise the complete case lifecycle with CPU model/environment doubles."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[1] / "benchmarks/static/cosmos_libero/run.py"


def entry():
    spec = importlib.util.spec_from_file_location("cosmos_case_lifecycle", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path):
    engines, simulators = [], []

    class Engine:
        def __init__(self, *args, **kwargs):
            self.metadata = {"backend": "test"}
            self.closed = False
            self.fail = False
            engines.append(self)

        def load(self, tasks, audit_path):
            assert tasks == ["instruction a", "instruction b"]
            audit_path.write_text('{"status":"passed"}')

        def reset(self, episode_id):
            self.episode_id = episode_id

        def infer_chunk(self, obs, task, sampling_seed):
            if self.fail:
                raise RuntimeError("inference failed")
            assert task in {"instruction a", "instruction b"}
            return [[index] * 7 for index in range(16)]

        def close(self):
            self.closed = True

    class Simulator:
        def __init__(self, task_id):
            self.task_name = "ab"[task_id]
            self.description = "instruction " + self.task_name
            self.control_dt = 0.05
            self.closed = False
            self.steps = 0
            self.actions = []
            simulators.append(self)

        def reset(self, initial, seed):
            self.steps = 0
            return {"state": 0}

        def step(self, action):
            self.actions.append(action[0])
            self.steps += 1
            return {"state": self.steps}, self.steps == 3, self.steps == 3

        def render(self):
            return self.steps

        def close(self):
            self.closed = True

    class Suite:
        def __init__(self, name):
            assert name == "libero_object"

        def tasks(self):
            return [
                {
                    "id": i,
                    "name": name,
                    "description": "instruction " + name,
                    "initial_states": 2,
                }
                for i, name in enumerate("ab")
            ]

        def make_simulator(self, task_id, **kwargs):
            return Simulator(task_id)

    plan = {
        "output_dir": str(tmp_path / "run"),
        "gpu": "3",
        "paper_async": {"schedule": "paper_async"},
        "resources": {
            key: str(tmp_path / key)
            for key in (
                "cosmos_source",
                "checkpoint",
                "dataset_stats",
                "text_embeddings",
                "vae_checkpoint",
                "libero_config_dir",
            )
        },
        "options": {
            "suite": "libero_object",
            "episodes": 3,
            "task_ids": None,
            "model_config": "test",
            "num_inference_steps": 5,
            "n_action_steps": 2,
            "max_steps": 3,
            "seed": 195,
            "env_seed": 0,
            "schedule": "paper_async",
            "overlap_actions": 2,
            "video_episodes_per_task": 0,
            "video_fps": 30,
        },
    }
    return plan, Engine, Suite, engines, simulators


def test_single_environment_case_writes_audited_results_and_summary(setup):
    plan, engine, suite, engines, simulators = setup
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    entry().execute(plan, engine_type=engine, suite_type=suite)
    output = Path(plan["output_dir"])
    rows = [
        json.loads(line)
        for line in (output / "episodes.jsonl").read_text().splitlines()
    ]
    assert [(row["task"], row["init_state_id"]) for row in rows] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
    ]
    assert all(
        row["primitive_steps"] == 3 and row["inference_calls"] == 2 for row in rows
    )
    requests = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    assert [request["history_offset_steps"] for request in requests] == [0, 2] * 3
    assert (
        json.loads((output / "case-manifest.json").read_text())["status"] == "completed"
    )
    assert json.loads((output / "coverage.json").read_text())["status"] == "passed"
    stats = json.loads((output / "episode-summary.json").read_text())["runs"][0][
        "statistics"
    ]["overall"]
    assert (
        stats["successes"] == 3
        and stats["failure_penalized"]["mean_control_steps"] == 3
    )
    assert "Success only" in (output / "run.log").read_text()
    assert all(resource.closed for resource in [*engines, *simulators])
    assert simulators[0].actions == [0, 1, 0] * 2
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


def test_infrastructure_error_does_not_become_a_failed_trial(setup):
    plan, engine, suite, engines, simulators = setup

    class Broken(engine):
        def load(self, tasks, audit_path):
            super().load(tasks, audit_path)
            self.fail = True

    with pytest.raises(RuntimeError, match="inference failed"):
        entry().execute(plan, engine_type=Broken, suite_type=suite)
    output = Path(plan["output_dir"])
    assert json.loads((output / "case-manifest.json").read_text())["status"] == "failed"
    assert json.loads((output / "coverage.json").read_text())["completed_episodes"] == 0
    assert (output / "episodes.jsonl").read_text() == ""
    assert not (output / "episode-summary.json").exists()
    assert all(resource.closed for resource in [*engines, *simulators])


def test_case_refuses_existing_output(setup):
    plan, engine, suite, *_ = setup
    output = Path(plan["output_dir"])
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        entry().execute(plan, engine_type=engine, suite_type=suite)
    assert marker.read_text() == "keep"


def test_episode_allocation_cannot_repeat_initial_states():
    tasks = [
        {"id": 0, "name": "a", "description": "a", "initial_states": 2},
        {"id": 1, "name": "b", "description": "b", "initial_states": 1},
    ]
    assert len(entry().allocate_episodes(tasks, 3)) == 3
    with pytest.raises(ValueError, match="exceed"):
        entry().allocate_episodes(tasks, 4)
