"""Real integer GEMM versus its quantized mathematical reference."""

import os
from pathlib import Path
import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)
import torch


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(1, 37, 130), (50, 80, 256)])
@pytest.mark.parametrize("tactic", [0, 1])
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
