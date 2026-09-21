"""Task ratios use observed matched pilots and keep modeled time explicit."""

import importlib
from pathlib import Path
import pytest


def api():
    assert (
        Path(__file__).parents[1] / "src/robotics_bench/profiling/task_report.py"
    ).exists(), "Task comparison is missing"
    return importlib.import_module("robotics_bench.profiling.task_report")


def row(success, steps, model_ms, host_s):
    return {
        "init_state_id": 0,
        "success": success,
        "primitive_steps": steps,
        "max_primitive_steps": 10,
        "paper_model_task_ms": model_ms,
        "host_task_seconds": host_s,
        "paper_model_failure_budget_ms": model_ms if success else 1000,
    }


def test_success_only_and_failure_budget_are_distinct():
    runs = [
        {
            "id": "original",
            "inference_ms": 10,
            "episodes": [
                row(True, 4, 100, 1),
                {**row(False, 10, 1000, 10), "init_state_id": 1},
            ],
        },
        {
            "id": "candidate",
            "inference_ms": 5,
            "episodes": [
                row(True, 6, 200, 2),
                {**row(False, 10, 1000, 10), "init_state_id": 1},
            ],
        },
    ]
    result = api().summarize(runs)
    assert result[1]["speedup_inference"] == 2
    assert result[1]["speedup_task_success_paper_model"] == 0.5
    assert result[1]["speedup_task_success_host"] == 0.5
    assert result[1]["mean_control_steps_failure_budget"] == 8
    assert result[1]["speedup_task_failure_budget_paper_model"] == pytest.approx(
        550 / 600
    )


def test_no_success_has_no_success_conditioned_speedup():
    result = api().summarize(
        [
            {"id": "original", "inference_ms": 10, "episodes": [row(True, 4, 100, 1)]},
            {
                "id": "candidate",
                "inference_ms": 5,
                "episodes": [row(False, 10, 1000, 10)],
            },
        ]
    )
    assert result[1]["success_rate"] == 0
    assert result[1]["speedup_task_success_paper_model"] is None
    assert result[1]["speedup_task_success_host"] is None


def test_rejects_different_initial_state_coverage():
    with pytest.raises(ValueError, match="coverage"):
        api().summarize(
            [
                {
                    "id": "original",
                    "inference_ms": 10,
                    "episodes": [row(True, 4, 100, 1)],
                },
                {
                    "id": "candidate",
                    "inference_ms": 5,
                    "episodes": [{**row(True, 4, 100, 1), "init_state_id": 1}],
                },
            ]
        )
