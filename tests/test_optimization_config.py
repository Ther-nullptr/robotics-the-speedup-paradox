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


def test_norm_pack_requires_integer_dit_and_preserves_bf16_reference():
    module = api()
    switches = ("modulation", "modulation_quant", "norm_modulation_quant")
    config = module.OptimizationConfig(switches, "int4", ("dit",))
    rows = dict(module.measurement_configurations(config))
    assert rows["optimized-bf16"].enabled == ("modulation",)
    combined = module.OptimizationConfig(
        (*switches, "gated_residual", "residual_norm_modulation_quant"),
        "int4",
        ("dit",),
    )
    assert dict(module.measurement_configurations(combined))[
        "optimized-bf16"
    ].enabled == (
        "modulation",
        "gated_residual",
    )
    with pytest.raises(ValueError, match="residual_norm_modulation_quant requires"):
        module.OptimizationConfig(("residual_norm_modulation_quant",), "int4", ("dit",))
    for options in (
        dict(switches=switches, precision="bf16", scopes=("dit",)),
        dict(switches=switches, precision="int4", scopes=("text",)),
        dict(switches=("norm_modulation_quant",), precision="int4", scopes=("dit",)),
    ):
        with pytest.raises(ValueError, match="norm_modulation_quant requires"):
            module.OptimizationConfig(**options)


def test_expanded_tactics_and_progressive_tier_roundtrip():
    config = api().OptimizationConfig(
        precision="int8", scopes=("dit",), tactic=7, quant_tier=3
    )
    from robotics_bench.optimizations.entry import from_dict

    assert from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        api().OptimizationConfig(quant_tier=1)


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
