"""Alternative backends remain independent shared-precision experiment settings."""

import pytest
from robotics_bench.optimizations.config import (
    OptimizationConfig,
    measurement_configurations,
)


def test_hotspot_variants_receive_matched_bf16_reference():
    base = OptimizationConfig(("modulation", "cuda_graph"), scopes=("dit",))
    switches = [
        "modulation",
        "cuda_graph",
        "vae_norm_fusion",
        "vae_silu_fusion",
        "attention_sdpa_flash",
    ]
    rows = measurement_configurations(
        base,
        [{"id": "int4-fused", "precision": "int4", "switches": switches}],
        allow_shared_variants=True,
    )
    assert rows[-2][1].precision == "bf16"
    assert rows[-2][1].enabled == rows[-1][1].enabled


@pytest.mark.parametrize(
    "switches,reason",
    [
        (("attention_sdpa_flash", "attention_flash_attn"), "attention"),
        (("conv_cutlass_0", "conv_cutlass_1"), "convolution"),
        (("vae_silu_fusion",), "vae_norm_fusion"),
    ],
)
def test_conflicting_hotspot_settings_are_rejected(switches, reason):
    with pytest.raises(ValueError, match=reason):
        OptimizationConfig(switches, scopes=("dit",))
