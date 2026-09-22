"""Cross-KV buffers refresh across prompts without stale CUDA graph constants."""

from contextlib import nullcontext
import gc
import os
import sys
from types import SimpleNamespace
import weakref

import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)

import torch

from robotics_bench.optimizations.cosmos_kv_cache import cache_cross_attention
from robotics_bench.optimizations.quantization import quantize_linears
from robotics_kernels.common.graph import CudaGraphCall


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_selfattn = False
        self.qkv_format = "bshd"
        self.n_heads, self.head_dim, self.context_dim = 16, 128, 1024
        self.q_proj = torch.nn.Linear(2048, 2048, bias=False)
        self.k_proj = torch.nn.Linear(1024, 2048, bias=False)
        self.v_proj = torch.nn.Linear(1024, 2048, bias=False)
        self.q_norm = torch.nn.RMSNorm(128, eps=1e-6)
        self.k_norm = torch.nn.RMSNorm(128, eps=1e-6)
        self.v_norm = torch.nn.Identity()
        self.attn_op = SimpleNamespace(pg=None)

    def compute_qkv(self, x, context=None, rope_emb=None):
        q = self.q_proj(x).reshape(*x.shape[:-1], 16, 128)
        k = self.k_proj(context).reshape(*context.shape[:-1], 16, 128)
        v = self.v_proj(context).reshape(*context.shape[:-1], 16, 128)
        return self.q_norm(q), self.k_norm(k), self.v_norm(v)

    def forward(self, x, context):
        q, k, v = self.compute_qkv(x, context)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return out.transpose(1, 2).flatten(2)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "robotics_bench.models.cosmos.minimal_v4_dit",
        SimpleNamespace(Attention=Attention),
    )
    model = torch.nn.Module()
    model.net = torch.nn.Module()
    model.net.blocks = torch.nn.ModuleList([Attention()])
    model.net.use_crossattn_projection = False
    model.net.extra_image_context_dim = None
    model.net.is_context_parallel_enabled = False
    return model.eval().cuda().to(torch.bfloat16)


@pytest.mark.parametrize("precision", ["bf16", "int8", "int4"])
@torch.inference_mode()
def test_graph_replay_a_b_a_keeps_values_current_and_cache_addresses_fixed(
    model, precision
):
    torch.manual_seed(71)
    x = torch.randn(1, 33, 2048, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1, 512, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    quantization = (
        nullcontext()
        if precision == "bf16"
        else quantize_linears(
            model.net,
            precision,
            scopes=("dit",),
            selector=lambda name: "dit",
            shared_quant=True,
            pack_reuse=True,
        )
    )
    with quantization:
        original = model.net.blocks[0].forward
        expected = [original(x, context).clone() for context in (a, b, a)]
        assert not torch.equal(expected[0], expected[1])
        with cache_cross_attention(model) as cache:
            graph = CudaGraphCall(original, retained_tensors=cache.retained_tensors)
            addresses = None
            for context, reference in zip((a, b, a), expected):
                with model._robotics_action_only_context():
                    for _ in range(5):
                        cache.prepare(context)
                        current = tuple(t.data_ptr() for t in cache.retained_tensors())
                        addresses = current if addresses is None else addresses
                        assert addresses == current
                        actual = graph(x, context)
                        assert torch.equal(actual, reference)
            assert graph.captures == cache.layout_generation == 1
            assert graph.replays == 15
            assert cache.report["refreshes"] == 3
            assert cache.report["mutation_checks"]["exact_tensor_compare"] == 12
            assert len(graph._constants) == 2


@torch.inference_mode()
def test_old_graph_retains_buffers_after_shape_change_and_cache_context_exit(model):
    x = torch.randn(1, 33, 2048, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1, 512, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, 256, 1024, device="cuda", dtype=torch.bfloat16)
    original = model.net.blocks[0].forward
    with cache_cross_attention(model) as cache:
        old_graph = CudaGraphCall(original, retained_tensors=cache.retained_tensors)
        with model._robotics_action_only_context():
            cache.prepare(a)
            old_graph(x, a)
        old_refs = tuple(weakref.ref(t) for t in cache.retained_tensors())
        with model._robotics_action_only_context():
            cache.prepare(b)
            assert cache.layout_generation == 2
            new_graph = CudaGraphCall(original, retained_tensors=cache.retained_tensors)
            new_graph(x, b)
        gc.collect()
        assert all(ref() is not None for ref in old_refs)
        assert all(
            a is not b for a, b in zip(old_graph._constants, new_graph._constants)
        )
    gc.collect()
    assert all(ref() is not None for ref in old_refs)
    del old_graph
    gc.collect()
    assert all(ref() is None for ref in old_refs)
