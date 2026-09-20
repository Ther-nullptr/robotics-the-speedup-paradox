"""Explicit opt-in CUDA numerical/stream tests; not part of public CPU CI."""

import os
from pathlib import Path

import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip(
        "Set ROBOTICS_GPU_TESTS=1 in the model environment", allow_module_level=True
    )

import torch


def api():
    assert (Path(__file__).parents[2] / "src/robotics_kernels/fused.py").exists(), (
        "Fused kernels are not implemented"
    )
    from robotics_kernels import fused

    return fused


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_preserves_each_rounding_boundary_on_current_stream(dtype):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q = torch.randn(2, 17, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
        k = torch.randn(2, 17, 1, 64, device="cuda", dtype=dtype).transpose(1, 2)
        cos = torch.randn(2, 17, 64, device="cuda", dtype=dtype)
        sin = torch.randn_like(cos)

        def reference(x):
            a, b = x.chunk(2, dim=-1)
            return x * cos[:, None] + torch.cat((-b, a), dim=-1) * sin[:, None]

        expected = (reference(q), reference(k))
        actual = api().rope(q, k, cos, sin)
    stream.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_residual_keeps_intermediate_rounding(dtype):
    x = torch.randn(2, 17, 256, device="cuda", dtype=dtype)
    y = torch.randn_like(x)
    gate = torch.randn(2, 1, 256, device="cuda", dtype=dtype)
    assert torch.equal(api().gated_residual(x, y, gate), x + y * gate)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gelu_mul_matches_torch_and_handles_tail(dtype):
    gate = torch.randn(3, 7, 258, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)
    expected = torch.nn.functional.gelu(gate, approximate="tanh") * up
    assert torch.equal(api().gelu_mul(gate, up), expected)


def test_norm_affine_preserves_fp32_scaling_and_bf16_output():
    x = torch.randn(2, 13, 256, device="cuda", dtype=torch.bfloat16)
    inv = torch.rsqrt(torch.mean(torch.square(x.float()), dim=-1, keepdim=True) + 1e-6)
    weight = torch.randn(256, device="cuda", dtype=torch.float32)
    expected = (x * inv * (1 + weight)).to(x.dtype)
    assert torch.equal(api().norm_affine(x, inv, weight), expected)
    modulation = torch.randn(2, 1, 768, device="cuda", dtype=torch.float32)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    expected = (x * inv * (1 + scale) + shift).to(x.dtype)
    assert torch.equal(api().norm_affine(x, inv, scale, shift), expected)


def test_adaln_modulation_preserves_bf16_intermediates():
    x = torch.randn(2, 17, 256, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 256, device="cuda", dtype=x.dtype)
    shift = torch.randn_like(scale)
    assert torch.equal(api().modulate(x, scale, shift), x * (1 + scale) + shift)
