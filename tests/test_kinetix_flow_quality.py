"""Flow-step sweeps keep paired seeds and the original native protocol."""

import importlib.util
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "benchmarks/dynamic/kinetix" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_jobs_cover_each_paired_episode_once_without_splitting_flow_counts():
    module = script("flow_quality.py")
    jobs = module.make_jobs(
        python="python",
        policy_dir="policies",
        output_dir="output",
        levels=list(module.LEVELS),
        flow_steps=[1, 2, 3, 4, 5],
        episodes=128,
        start_seed=0,
    )
    keys = []
    for job in jobs:
        args = job["args"]

        def value(key):
            return args[args.index(key) + 1]

        assert value("--latencies-ms") == "0"
        assert value("--execute-horizon") == "4"
        assert value("--action-noise-std") == "0.1"
        assert "--max-steps" not in args and "--control-hz" not in args
        for flow in map(int, value("--flow-steps").split(",")):
            for seed in range(
                int(value("--start-seed")),
                int(value("--start-seed")) + int(value("--episodes")),
            ):
                keys.append((job["task"], flow, seed))
    assert len(jobs) == 12
    assert len(keys) == len(set(keys)) == 7680


@pytest.mark.parametrize(
    "field,value",
    [("levels", ["car_launch", "car_launch"]), ("flow_steps", [1, 1]), ("episodes", 0)],
)
def test_invalid_sweep_cannot_silently_duplicate_or_drop_cells(field, value):
    module = script("flow_quality.py")
    args = dict(
        python="python",
        policy_dir="policies",
        output_dir="output",
        levels=["car_launch"],
        flow_steps=[1, 5],
        episodes=2,
        start_seed=0,
    )
    args[field] = value
    with pytest.raises(ValueError):
        module.make_jobs(**args)
