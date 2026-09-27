"""Calibration isolation and deployment metrics for latency studies."""

import importlib.util
from pathlib import Path

import pytest


def load_analysis():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/kinetix/analyze_latency_study.py"
    )
    spec = importlib.util.spec_from_file_location("kinetix_latency_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calibration():
    cells = [
        {
            "role": "quality",
            "flow_steps": n,
            "successes": success,
            "episodes": 10,
            "effective_latency_ms": 0,
            "mode": "fine",
        }
        for n, success in enumerate([5, 6, 8, 7, 8], start=1)
    ]
    for mode, points in [
        ("fine", [(0, 8), (1000 / 60, 7), (1000 / 30, 5), (50, 4), (2000 / 30, 2)]),
        ("coarse", [(0, 8), (1000 / 30, 5), (2000 / 30, 2)]),
    ]:
        cells.extend(
            {
                "role": "calibration",
                "flow_steps": 5,
                "successes": count,
                "episodes": 10,
                "effective_latency_ms": delay,
                "mode": mode,
            }
            for delay, count in points
        )
    return cells


def test_heldout_outcomes_cannot_change_calibration_model():
    analysis = load_analysis()
    cells = calibration()
    expected = analysis.fit_task_model(cells)
    cells.append(
        {
            "role": "validation",
            "successes": 0,
            "episodes": 10000,
            "flow_steps": 5,
            "effective_latency_ms": 1,
            "mode": "fine",
        }
    )
    assert analysis.fit_task_model(cells) == expected
    assert expected["quality_probabilities"] == [0.5, 0.6, 0.8, 0.7, 0.8]


def test_coarse_predictions_use_effective_bins_and_fine_preserves_delay():
    analysis = load_analysis()
    model = analysis.fit_task_model(calibration())
    assert analysis.predict_success(model, "coarse", 5, 20) == pytest.approx(0.5)
    assert analysis.predict_success(model, "coarse", 5, 40) == pytest.approx(0.5)
    assert analysis.predict_success(model, "fine", 5, 20) != pytest.approx(
        analysis.predict_success(model, "fine", 5, 40)
    )


def test_validation_rmse_does_not_count_reused_calibration_cells():
    analysis = load_analysis()
    cells = [
        {
            "flow_steps": 1,
            "success_rate": 0.0,
            "predicted_success": 1.0,
            "calibration_overlap": True,
        },
        {
            "flow_steps": 2,
            "success_rate": 0.8,
            "predicted_success": 0.7,
            "calibration_overlap": False,
        },
        {
            "flow_steps": 3,
            "success_rate": 0.8,
            "predicted_success": 0.75,
            "calibration_overlap": False,
        },
    ]
    metrics = analysis.validation_metrics(cells)
    assert metrics["heldout_cells"] == 2
    assert metrics["heldout_rmse"] == pytest.approx((0.1**2 / 2 + 0.05**2 / 2) ** 0.5)
    assert metrics["observed_best_integers"] == [2, 3]
    assert metrics["predicted_best_integer"] == 1
