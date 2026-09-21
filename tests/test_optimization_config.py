"""CPU contracts for explicit optimization switches and comparison evidence."""

import importlib
import math
from pathlib import Path

import pytest


def api():
    path = Path(__file__).parents[1] / "src/robotics_bench/optimizations/config.py"
    assert path.exists(), "Optimization configuration is not implemented"
    return importlib.import_module("robotics_bench.optimizations.config")


def test_defaults_preserve_original_path():
    config = api().OptimizationConfig()
    assert config.enabled == ()
    assert config.precision == "bf16"
    assert config.to_dict()["switches"] == []


def test_independent_switches_and_unknowns():
    config = api().OptimizationConfig(switches=("flow_loop", "rope"))
    assert config.enabled == ("flow_loop", "rope")
    with pytest.raises(ValueError, match="Unknown"):
        api().OptimizationConfig(switches=("made_up",))
    with pytest.raises(ValueError, match="Duplicate"):
        api().OptimizationConfig(switches=("rope", "rope"))


def test_comparison_uses_direct_baseline_and_retains_quality_failure():
    compare = api().compare_samples
    result = compare([100, 110, 90], [50, 55, 45], exact=False, max_abs=0.1)
    assert result["speedup_vs_baseline"] == 2
    assert result["saved_ms"] == 50
    assert result["exact"] is False
    assert result["baseline_samples_ms"] == [100, 110, 90]


@pytest.mark.parametrize("samples", [[], [0], [-1], [math.nan], [math.inf], [True]])
def test_reject_invalid_timing(samples):
    with pytest.raises(ValueError):
        api().compare_samples([1], samples, exact=True, max_abs=0)
