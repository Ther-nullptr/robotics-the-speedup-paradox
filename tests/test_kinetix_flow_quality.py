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


def test_speed_gate_requires_full_state_and_action_traces():
    import copy

    equivalent = script("benchmark_eval.py").equivalent_trace
    trace = {
        "result": {"primitive_steps": 1, "inference_calls": 1},
        "state_hashes": ["state"],
        "action_hashes": ["action"],
        "events": [
            {"kind": "inference", "host_policy_call_seconds": 1.0},
            {"kind": "control", "reward": 0.0},
        ],
    }
    candidate = copy.deepcopy(trace)
    candidate["events"][0]["host_policy_call_seconds"] = 2.0
    assert equivalent(trace, candidate)
    candidate["events"][1]["reward"] = 1.0
    assert not equivalent(trace, candidate)
    candidate["events"][1]["reward"] = 0.0
    candidate["state_hashes"] = ["different-state"]
    assert not equivalent(trace, candidate)
    candidate["state_hashes"] = []
    assert not equivalent(trace, candidate)
    missing = copy.deepcopy(trace)
    missing["events"] = []
    assert not equivalent(missing, missing)


def test_speedup_is_withheld_when_timed_outcomes_differ_from_verified_traces():
    import copy

    compare = script("benchmark_eval.py").comparison_summary
    trace = {
        "result": {"primitive_steps": 1, "inference_calls": 1},
        "state_hashes": ["state"],
        "action_hashes": ["action"],
        "events": [{"kind": "inference"}, {"kind": "control"}],
    }
    report = {
        "seeds": [0],
        "repeats": 1,
        "equivalence_traces": {"reference": [trace], "preprocess_jit": [trace]},
        "rows": [
            {
                "mode": mode,
                "seed": 0,
                "repeat": 0,
                "result": copy.deepcopy(trace["result"]),
                "wall_seconds": wall,
            }
            for mode, wall in (("reference", 2.0), ("preprocess_jit", 1.0))
        ],
    }
    assert compare(report)["validated_speedup"] == 2.0
    report["rows"][1]["result"]["primitive_steps"] = 0
    assert compare(report)["validated_speedup"] is None
    # Even matching timed outcomes must describe the verified episode.
    report["rows"][0]["result"]["primitive_steps"] = 0
    assert compare(report)["validated_speedup"] is None
    report["rows"] = report["rows"][:1]
    assert compare(report)["validated_speedup"] is None
