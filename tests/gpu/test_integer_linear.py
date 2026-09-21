"""Real integer GEMM versus its quantized mathematical reference."""

import os
from pathlib import Path
import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)
import torch


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(1, 37, 130), (50, 80, 256)])
@pytest.mark.parametrize("tactic", range(8))
def test_integer_linear_matches_quantized_reference_on_nondefault_stream(
    bits, shape, tactic
):
    assert (Path(__file__).parents[2] / "src/robotics_kernels/integer.py").exists(), (
        "Integer backend is not implemented"
    )
    from robotics_kernels.integer import IntegerLinear

    m, n, k = shape
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        original = torch.nn.Linear(k, n, device="cuda", dtype=torch.bfloat16).eval()
        layer = IntegerLinear.from_linear(original, bits=bits, tactic=tactic)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        actual = layer(x)
        limit = 2 ** (bits - 1) - 1
        ws = original.weight.float().abs().amax(dim=1).clamp_min(1e-30) / limit
        xs = x.float().abs().amax(dim=1).clamp_min(1e-30) / limit
        qw = torch.round(original.weight.float() / ws[:, None]).clamp(-limit, limit)
        qx = torch.round(x.float() / xs[:, None]).clamp(-limit, limit)
        ref = ((qx @ qw.T) * xs[:, None] * ws[None, :] + original.bias.float()).to(
            x.dtype
        )
    stream.synchronize()
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    assert layer.backend == f"cutlass_sm80_int{bits}_s32_bf16"
    assert layer.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("bits", [4, 8])
def test_fused_gelu_pack_matches_separate_integer_path(bits):
    from robotics_kernels.integer import IntegerLinear

    layer = IntegerLinear.from_linear(
        torch.nn.Linear(256, 80, device="cuda", dtype=torch.bfloat16).eval(), bits=bits
    )
    gate = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    expected = layer(torch.nn.functional.gelu(gate, approximate="tanh") * up)
    assert hasattr(layer, "forward_gelu"), "Fused activation preparation is missing"
    assert torch.equal(layer.forward_gelu(gate, up), expected)


@pytest.mark.parametrize("bits", [4, 8])
def test_grouped_integer_projections_preserve_padded_outputs_and_refresh_input(bits):
    from robotics_kernels.ampere_ada import integer

    assert hasattr(integer, "IntegerProjectionGroup"), (
        "Grouped integer projections are missing"
    )
    layers = [
        integer.IntegerLinear.from_linear(
            torch.nn.Linear(
                130, n, bias=i != 1, device="cuda", dtype=torch.bfloat16
            ).eval(),
            bits=bits,
        )
        for i, n in enumerate((37, 17, 48))
    ]
    group = integer.IntegerProjectionGroup(layers)
    for rows in (9, 5):
        x = torch.randn(rows, 130, device="cuda", dtype=torch.bfloat16)
        expected = [layer(x) for layer in layers]
        actual = [group.project(x, i) for i in range(len(layers))]
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert all(a.is_contiguous() for a in actual)
        assert group._input is None and group._output is None


def test_group_rejects_mixed_integer_formats():
    from robotics_kernels.ampere_ada import integer

    assert hasattr(integer, "IntegerProjectionGroup"), (
        "Grouped integer projections are missing"
    )
    original = torch.nn.Linear(128, 32, device="cuda", dtype=torch.bfloat16).eval()
    layers = [integer.IntegerLinear.from_linear(original, bits=b) for b in (4, 8)]
    with pytest.raises(ValueError, match="format"):
        integer.IntegerProjectionGroup(layers)


def test_int4_register_pack_matches_reference_and_fused_gelu():
    from robotics_kernels.ampere_ada.integer import IntegerLinear

    original = torch.nn.Linear(258, 37, device="cuda", dtype=torch.bfloat16).eval()
    optimized = IntegerLinear.from_linear(original, bits=4, pack_reuse=True)
    reference = IntegerLinear.from_linear(original, bits=4)
    gate = torch.randn(19, 258, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    assert torch.equal(optimized(gate), reference(gate))
    assert torch.equal(
        optimized.forward_gelu(gate, up), reference.forward_gelu(gate, up)
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_group_views_feed_fused_quantization_without_contiguous_copies(bits):
    from robotics_kernels.ampere_ada.integer import (
        IntegerLinear,
        IntegerProjectionGroup,
    )

    layers = [
        IntegerLinear.from_linear(
            torch.nn.Linear(128, 258, device="cuda", dtype=torch.bfloat16), bits=bits
        )
        for _ in range(2)
    ]
    group = IntegerProjectionGroup(layers, contiguous_outputs=False)
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16)
    gate, up = group.project(x, 0), group.project(x, 1)
    assert not gate.is_contiguous() and not up.is_contiguous()
    assert gate.untyped_storage().data_ptr() == up.untyped_storage().data_ptr()
    down = IntegerLinear.from_linear(
        torch.nn.Linear(258, 37, device="cuda", dtype=torch.bfloat16),
        bits=bits,
        pack_reuse=True,
    )
    assert torch.equal(down(gate), down(gate.contiguous()))
    assert torch.equal(
        down.forward_gelu(gate, up),
        down.forward_gelu(gate.contiguous(), up.contiguous()),
    )
