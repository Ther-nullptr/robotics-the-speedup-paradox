"""Numeric diagnostics distinguish changed values from signed-zero bits."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/kinetix/diagnose_native_repeatability.py"
    )
    spec = importlib.util.spec_from_file_location("native_repeatability_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@dataclass
class State:
    velocity: object


def test_reports_named_first_numeric_change_and_signed_zero_separately():
    baseline = State(np.array([1.0, 0.0], dtype=np.float32))
    candidate = State(
        np.array([np.nextafter(np.float32(1), np.float32(2)), -0.0], dtype=np.float32)
    )
    diff = module().state_differences(baseline, candidate)
    assert len(diff) == 1
    assert diff[0]["path"] == "velocity"
    assert diff[0]["bitwise_different_elements"] == 2
    assert diff[0]["numerically_different_elements"] == 1
    assert diff[0]["signed_zero_differences"] == 1
    assert diff[0]["max_absolute_difference"] == float(np.finfo(np.float32).eps)
    assert module().state_differences(baseline, baseline) == []


def test_different_state_layout_is_rejected_instead_of_broadcast():
    with pytest.raises(ValueError, match="layout"):
        module().state_differences(State(np.zeros((1, 2))), State(np.zeros(2)))


def test_snapshot_restore_checks_layout_and_trace_comparison_checks_termination():
    diagnostic = module()
    state = State(np.array([1, 2], dtype=np.float32))
    restored = diagnostic.restore_arrays(state, diagnostic.named_arrays(state))
    assert diagnostic.state_differences(state, restored) == []
    with pytest.raises(ValueError, match="layout"):
        diagnostic.restore_arrays(state, {"velocity": np.zeros(3, dtype=np.float32)})
    left = {"state_hashes": ["a", "b"], "result": {"success": True}}
    right = {"state_hashes": ["a", "b", "c"], "result": {"success": False}}
    result = diagnostic.compare_traces(left, right)
    assert result["first_state_difference"] is None
    assert not result["same_length"] and not result["same_result"]
