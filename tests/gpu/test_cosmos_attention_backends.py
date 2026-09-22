"""Explicit attention backends: numerical checks, capture, and failure boundaries."""

import os
import sys
from types import SimpleNamespace

import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from robotics_bench.optimizations.cosmos_attention import (
    AttentionOperator,
    cosmos_attention_backend,
)


@pytest.mark.parametrize(
    "backend", ["sdpa_flash", "sdpa_efficient", "sdpa_math", "flash_attn"]
)
@pytest.mark.parametrize("lengths,head_dim", [((33, 33), 32), ((97, 51), 128)])
@torch.inference_mode()
def test_dense_backends_match_math_for_self_and_cross_attention(
    backend, lengths, head_dim
):
    torch.manual_seed(42)
    qlen, klen = lengths
    # Non-contiguous head/row strides occur around projection and RoPE paths.
    q = torch.randn(2, qlen, 8, head_dim, device="cuda", dtype=torch.bfloat16)[
        :, :, ::2, :
    ]
    k = torch.randn(2, klen, 8, head_dim, device="cuda", dtype=torch.bfloat16)[
        :, :, ::2, :
    ]
    v = torch.randn_like(k)
    scale = head_dim**-0.5 * 0.75
    expected = AttentionOperator("sdpa_math")(q, k, v, scale=scale)
    actual = AttentionOperator(backend)(q, k, v, scale=scale)
    assert actual.shape == (2, qlen, 4 * head_dim)
    assert actual.dtype == q.dtype
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.02)


@pytest.mark.parametrize("backend", ["sdpa_flash", "sdpa_efficient", "flash_attn"])
@torch.inference_mode()
def test_backend_supports_capture_and_counts_python_invocations(backend):
    q = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    op = AttentionOperator(backend)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            expected = op(q, q, q)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op(q, q, q)
    calls_after_capture = op.record["python_calls"]
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
    assert op.record["python_calls"] == calls_after_capture


@torch.inference_mode()
def test_flash_attn_rejects_mask_and_non_square_causal_instead_of_changing_semantics():
    q = torch.randn(1, 32, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    op = AttentionOperator("flash_attn")
    with pytest.raises(ValueError, match="explicit attention mask"):
        op(q, k, k, attn_mask=torch.ones(32, 64, device="cuda", dtype=torch.bool))
    with pytest.raises(ValueError, match="causal alignment"):
        op(q, k, k, is_causal=True)
    assert op.record["python_calls"] == 0


@torch.inference_mode()
def test_sdpa_backend_choice_is_restored_even_when_backend_rejects_shape():
    # head_dim > 256 is unsupported by PyTorch FlashAttention in the test stack.
    q = torch.randn(1, 32, 1, 384, device="cuda", dtype=torch.bfloat16)
    op = AttentionOperator("sdpa_flash")
    with sdpa_kernel(SDPBackend.MATH):
        with pytest.raises(RuntimeError):
            op(q, q, q)
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
    assert op.record["python_calls"] == 0


@pytest.mark.parametrize(
    "backend", ["sdpa_flash", "sdpa_efficient", "sdpa_math", "flash_attn"]
)
@torch.inference_mode()
def test_native_minimal_a2a_preserves_bf16_conversion_and_registered_module(
    backend, monkeypatch
):
    native = pytest.importorskip(
        "cosmos_policy._src.predict2.networks.a2a_cp",
        reason="This integration test needs the case's Cosmos source on PYTHONPATH",
    )

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = "minimal_a2a"
            self.qkv_format = "bshd"
            self.is_selfattn = False
            self.n_heads = 4
            self.head_dim = 128
            self.attn_op = native.MinimalA2AAttnOp()

    # Exercise the actual upstream A2A operator without requiring a checkpoint
    # or constructing the full owned DiT and all of its framework dependencies.
    monkeypatch.setitem(
        sys.modules,
        "robotics_bench.models.cosmos.minimal_v4_dit",
        SimpleNamespace(Attention=Attention),
    )
    network = torch.nn.Sequential(Attention()).eval()
    original = network[0].attn_op
    assert original.pg is None
    q = torch.randn(1, 97, 4, 128, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 51, 4, 128, device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)
    expected = original(q, k, v)
    with cosmos_attention_backend(network, backend) as report:
        actual = original(q, k, v)
        assert network[0].attn_op is original
        assert actual.dtype == expected.dtype == torch.bfloat16
        assert actual.shape == expected.shape == (1, 97, 512)
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.02)
        assert report["coverage"]["0"]["context_parallel_size"] == 1
        assert report["coverage"]["0"]["python_calls"] == 1
        assert (
            report["coverage"]["0"]["observed_shapes"][0]["dtype"] == "torch.bfloat16"
        )
    assert "forward" not in vars(original)
    assert torch.equal(original(q, k, v), expected)
