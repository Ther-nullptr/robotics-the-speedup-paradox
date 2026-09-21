"""Cosmos modulation packing preserves the quantized mathematical reference."""

import os
from pathlib import Path
import importlib
import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)
import torch


def api():
    assert (
        Path(__file__).parents[2] / "src/robotics_kernels/ampere_ada/modulation.py"
    ).exists(), "Fused modulation packing is missing"
    return importlib.import_module("robotics_kernels.ampere_ada.modulation")


@pytest.mark.parametrize("bits", [4, 8])
def test_modulation_pack_preserves_bf16_rounding_and_padding(bits):
    from robotics_kernels.ampere_ada.integer import prepare_activation

    module = api()
    x = torch.randn(2, 17, 258, device="cuda", dtype=torch.bfloat16)
    parameters = torch.randn(2, 1, 3 * 258, device="cuda", dtype=x.dtype)
    scale, shift, _ = parameters.chunk(3, dim=-1)
    expected = prepare_activation(x * (1 + scale) + shift, bits, pack_reuse=True)
    actual = module.prepare_modulation(x, scale, shift, bits)
    assert torch.equal(actual.data, expected.data)
    assert torch.equal(actual.scales, expected.scales)
    assert actual.shape == expected.shape


def test_mixed_packed_inputs_reach_attention_and_mlp_linear_shapes():
    from robotics_bench.optimizations.cosmos import _modulation
    from robotics_kernels.ampere_ada.integer import IntegerLinear

    module = api()
    optimizer = importlib.import_module("robotics_bench.optimizations.cosmos")
    x = torch.randn(1, 2, 3, 4, 128, device="cuda", dtype=torch.bfloat16)
    norm = torch.nn.LayerNorm(128, elementwise_affine=False, device="cuda").eval()
    scale = torch.randn(1, 2, 1, 1, 128, device="cuda", dtype=x.dtype)
    shift = torch.randn_like(scale)
    reference = norm(x) * (1 + scale) + shift
    norm._robotics_quant_bits = (4, 8)
    packed = _modulation(x, norm, scale, shift, True)
    assert isinstance(packed, module.PackedActivations)
    attention = optimizer._attention_input(packed, (1, 24))
    for bits in (4, 8):
        linear = IntegerLinear.from_linear(
            torch.nn.Linear(128, 64, device="cuda", dtype=x.dtype).eval(), bits=bits
        )
        assert torch.equal(linear(packed), linear(reference))
        assert torch.equal(linear(attention), linear(reference.reshape(1, 24, 128)))


def test_packed_carrier_rejects_missing_format_and_changed_row_count():
    module = api()
    x = torch.randn(1, 17, 128, device="cuda", dtype=torch.bfloat16)
    scale = torch.zeros(1, 1, 128, device="cuda", dtype=x.dtype)
    packed = module.PackedActivations(
        {4: module.prepare_modulation(x, scale, scale, 4)}
    )
    with pytest.raises(ValueError, match="format"):
        packed.for_bits(8)
    with pytest.raises(ValueError, match="row count"):
        packed.with_leading_shape((1, 18))
