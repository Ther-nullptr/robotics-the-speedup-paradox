"""An evaluation speed summary cannot hide missing or divergent tasks."""

import copy
import importlib.util
from pathlib import Path

import pytest


def analyzer():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/kinetix/analyze_eval_benchmarks.py"
    )
    spec = importlib.util.spec_from_file_location("eval_analysis_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze


def benchmark(task, reference_seconds, candidate_seconds):
    trace = {
        "result": {"primitive_steps": 1, "inference_calls": 1, "success": True},
        "state_hashes": ["state"],
        "action_hashes": ["action"],
        "events": [{"kind": "inference"}, {"kind": "control"}],
    }
    return {
        "level": task,
        "status": "completed",
        "compare_preprocess": True,
        "flow_steps": 5,
        "seeds": [0],
        "repeats": 1,
        "scope": "native_batch1_runner_without_file_logging_or_video",
        "xla_flags": "",
        "identity": {
            "source_sha256": {"benchmark_eval.py": "entry", "environment.py": "native"},
            "python": "3.11",
            "packages": {"jax": "0.4.35"},
            "gpu_query": "0, uuid, RTX 6000 Ada, driver, 49140",
            "checkpoint": {"sha256": task},
        },
        "policy_metadata": {"parameter_dtypes": ["float32"]},
        "equivalence_traces": {
            "reference": [trace],
            "preprocess_jit": [copy.deepcopy(trace)],
        },
        "rows": [
            {
                "mode": mode,
                "repeat": 0,
                "seed": 0,
                "result": trace["result"],
                "wall_seconds": wall,
                "components": {
                    "environment_step_seconds": wall * 0.9,
                    "policy_call_seconds": wall * 0.05,
                },
            }
            for mode, wall in (
                ("reference", reference_seconds),
                ("preprocess_jit", candidate_seconds),
            )
        ],
    }


def test_aggregate_is_ratio_of_total_work_and_not_mean_of_task_speedups():
    report = analyzer()([benchmark("a", 2, 1), benchmark("b", 4, 4)], levels=["a", "b"])
    assert report["aggregate"]["total_rollout_speedup"] == pytest.approx(6 / 5)
    assert report["aggregate"]["equal_task_geometric_mean_speedup"] == pytest.approx(
        2**0.5
    )
    assert report["aggregate"]["timed_episode_pairs"] == 2


def test_failed_equivalence_is_retained_and_withholds_the_aggregate():
    bad = benchmark("b", 4, 1)
    bad["equivalence_traces"]["preprocess_jit"][0]["state_hashes"] = ["different"]
    report = analyzer()([benchmark("a", 2, 1), bad], levels=["a", "b"])
    assert report["complete"]
    assert not report["all_equivalent"]
    assert report["aggregate"] is None
    assert report["tasks"][1]["validated_speedup"] is None


def test_partial_duplicate_and_incompatible_tasks_do_not_produce_full_results():
    analyze = analyzer()
    a, b = benchmark("a", 2, 1), benchmark("b", 4, 4)
    with pytest.raises(ValueError, match="Incomplete"):
        analyze([a], levels=["a", "b"])
    assert analyze([a], levels=["a", "b"], allow_partial=True)["aggregate"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        analyze([a, a], levels=["a", "b"])
    b["identity"]["source_sha256"]["environment.py"] = "modified"
    with pytest.raises(ValueError, match="different"):
        analyze([a, b], levels=["a", "b"])
