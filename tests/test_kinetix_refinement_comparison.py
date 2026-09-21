"""Research comparisons must distinguish factors and transient trajectory drift."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "benchmarks/dynamic/kinetix" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows():
    return {
        "rows": [
            {"task": "example", "factor": factor, "variant": variant}
            for factor, variant in (
                (1, "direct"),
                (2, "impulse_motor_collision"),
                (4, "impulse_motor_collision"),
            )
        ]
    }


def test_explicit_factors_select_both_refinements_with_one_native_reference():
    compare = script("compare_refinement_videos.py")
    selected = compare.select_rows(
        rows(), "example", ["impulse_motor_collision"], [2, 4]
    )
    assert [r["factor"] for r in selected] == [1, 2, 4]
    with pytest.raises(ValueError, match="one refined row"):
        compare.select_rows(rows(), "example", ["impulse_motor_collision"], [8])


def test_ambiguous_factor_still_requires_explicit_selection():
    compare = script("compare_refinement_videos.py")
    with pytest.raises(ValueError, match="one refined row"):
        compare.select_rows(rows(), "example", ["impulse_motor_collision"])


def test_whole_prefix_metrics_detect_drift_even_when_final_frames_match():
    compare = script("research_timestep.py").compare
    reference = {
        "position": np.zeros((3, 1, 2)),
        "rotation": np.array([[0], [np.pi - 0.01], [0]]),
    }
    candidate = {
        "position": np.array([[[0, 0]], [[1, 0]], [[0, 0]]]),
        "rotation": np.array([[0], [-np.pi + 0.01], [0]]),
    }
    metrics = compare(reference, candidate)
    assert metrics["position_rmse_at_common_end"] == 0
    assert metrics["position_rmse_over_common_prefix"] == pytest.approx(0.5)
    assert metrics["position_max_body_distance_over_common_prefix"] == 1
    assert metrics["rotation_rmse_over_common_prefix"] == pytest.approx(
        0.02 / np.sqrt(2)
    )


def test_matching_solver_budget_preserves_iteration_count_without_rounding():
    iterations = script("refinement.py").solver_iterations_at_substep
    assert iterations(10, 2) * 2 == 10
    assert iterations(10, 1) == 10
    with pytest.raises(ValueError, match="divide"):
        iterations(10, 4)
