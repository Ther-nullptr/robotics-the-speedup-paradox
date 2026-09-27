"""Plotting must accept floating-point Wilson endpoints without hiding bad data."""

import importlib.util
from pathlib import Path

import pytest
from robotics_bench.dynamicvla_dom.analysis import wilson


def module(name):
    path = (
        Path(__file__).resolve().parents[1] / "benchmarks/dynamic/dynamicvla_dom" / name
    )
    spec = importlib.util.spec_from_file_location("_plot_test_" + path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_wilson_endpoint_roundoff_produces_nonnegative_errorbars():
    plot = module("plot_study.py")
    metrics = []
    for successes in (0, 5, 10):
        lo, hi = wilson(successes, 10)
        metrics.append(
            dict(
                success_rate=successes / 10,
                success_rate_wilson95_low=lo,
                success_rate_wilson95_high=hi,
            )
        )
    errors = plot.success_rate_errors(metrics)
    assert all(value >= 0 for row in errors for value in row)
    assert errors[1][-1] == 0
    with pytest.raises(ValueError, match="interval"):
        plot.success_rate_errors(
            [
                dict(
                    success_rate=0.5,
                    success_rate_wilson95_low=0.6,
                    success_rate_wilson95_high=0.7,
                )
            ]
        )


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_optional_plot_failure_is_recorded_without_stopping_evaluation(
    monkeypatch, failure
):
    import subprocess

    dense = module("dense_study.py")

    def fail(*args, **kwargs):
        assert kwargs["timeout"] == 120
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 120)
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(dense.subprocess, "run", fail)
    state = {}
    dense.render_progress(["python", "plot.py"], state, "block-00")
    assert state["plot_status"] == "failed"
    assert state["plot_errors"][0]["block"] == "block-00"
    assert state["plot_errors"][0]["error"]
