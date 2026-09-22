"""Attention alternatives preserve the owned dense-attention contract."""

import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


def api():
    path = (
        Path(__file__).parents[1]
        / "src/robotics_bench/optimizations/cosmos_attention.py"
    )
    assert path.exists(), "Explicit Cosmos attention backends are missing"
    return importlib.import_module("robotics_bench.optimizations.cosmos_attention")


def original_attention(*args, **kwargs):
    return "original"


class Attention:
    def __init__(self, backend="i4", **kwargs):
        self.backend = backend
        self.qkv_format = "bshd"
        self.training = False
        self.is_selfattn = True
        self.head_dim = 8
        self.n_heads = 2
        self.attn_op = original_attention
        vars(self).update(kwargs)


class Network:
    training = False

    def __init__(self, *modules):
        self.modules = modules

    def named_modules(self):
        yield "", self
        yield from ((f"blocks.{i}.self_attn", m) for i, m in enumerate(self.modules))


@pytest.fixture
def owned_attention(monkeypatch):
    dit = SimpleNamespace(
        Attention=Attention,
        i4_attention_op=original_attention,
        torch_attention_op=original_attention,
        flex_attention_op=original_attention,
    )
    monkeypatch.setitem(sys.modules, "robotics_bench.models.cosmos.minimal_v4_dit", dit)
    return dit


def test_unknown_backend_fails_before_importing_optional_runtime():
    module = api()
    with pytest.raises(ValueError, match="Unknown attention backend"):
        module.AttentionOperator("auto")


def test_instance_backend_change_restores_after_failure(owned_attention, monkeypatch):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    first, second = Attention(), Attention()
    with pytest.raises(RuntimeError, match="inference failed"):
        with module.cosmos_attention_backend(Network(first), "sdpa_flash") as report:
            assert isinstance(first.attn_op, module.AttentionOperator)
            assert second.attn_op is original_attention
            assert report["backend"] == "sdpa_flash"
            assert report["adapter"] == "robotics_bench.optimizations.cosmos_attention"
            assert report["coverage"]["blocks.0.self_attn"]["original_backend"] == "i4"
            raise RuntimeError("inference failed")
    assert first.attn_op is original_attention
    assert first.backend == "i4"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"training": True}, "inference-only"),
        ({"qkv_format": "sbhd"}, "bshd"),
        ({"backend": "unknown"}, "Unsupported original"),
        ({"attn_op": lambda *args: None}, "Unsupported attention operator"),
    ],
)
def test_unsupported_module_does_not_leave_partial_patches(
    owned_attention, monkeypatch, kwargs, message
):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    first, bad = Attention(), Attention(**kwargs)
    with pytest.raises(ValueError, match=message):
        with module.cosmos_attention_backend(Network(first, bad), "sdpa_flash"):
            pass
    assert first.attn_op is original_attention


def test_zero_matched_modules_is_an_error(owned_attention, monkeypatch):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    with pytest.raises(ValueError, match="No owned Cosmos attention"):
        with module.cosmos_attention_backend(Network(), "sdpa_math"):
            pass


def test_class_migrated_model_accepts_its_reviewed_upstream_function(
    owned_attention, monkeypatch
):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())

    def upstream_attention(*args):
        return "upstream"

    monkeypatch.setitem(
        sys.modules,
        "cosmos_policy._src.predict2.networks.minimal_v4_dit",
        SimpleNamespace(torch_attention_op=upstream_attention),
    )
    attention = Attention(backend="torch", attn_op=upstream_attention)
    with module.cosmos_attention_backend(Network(attention), "sdpa_flash"):
        assert isinstance(attention.attn_op, module.AttentionOperator)
    assert attention.attn_op is upstream_attention


def test_unknown_call_arguments_cannot_be_silently_ignored(monkeypatch):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    operator = module.AttentionOperator("sdpa_flash")
    with pytest.raises(TypeError, match="kv_cache_cfg"):
        operator(None, None, None, kv_cache_cfg=object())


class MinimalA2AAttnOp:
    __module__ = "cosmos_policy._src.predict2.networks.a2a_cp"

    def __init__(self, **kwargs):
        self.training = False
        self.pg = None
        self.stream = None
        self.local_attn = original_attention
        vars(self).update(kwargs)

    def forward(self, q, k, v):
        return "original A2A"

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@pytest.fixture
def a2a_definition(owned_attention, monkeypatch):
    owned_attention.MinimalA2AAttnOp = MinimalA2AAttnOp
    monkeypatch.setitem(
        sys.modules,
        "cosmos_policy._src.predict2.networks.a2a_cp",
        SimpleNamespace(
            MinimalA2AAttnOp=MinimalA2AAttnOp, attention=original_attention
        ),
    )


def test_single_device_minimal_a2a_keeps_module_and_restores_forward(
    a2a_definition, monkeypatch
):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    original = MinimalA2AAttnOp()
    attention = Attention(backend="minimal_a2a", attn_op=original)
    with module.cosmos_attention_backend(Network(attention), "sdpa_flash") as report:
        assert attention.attn_op is original
        assert "forward" in vars(original)
        coverage = report["coverage"]["blocks.0.self_attn"]
        assert coverage["original_backend"] == "minimal_a2a"
        assert coverage["context_parallel_size"] == 1
        assert coverage["context_parallel_group"] is None
        assert coverage["input_conversion"] == "bfloat16"
    assert attention.attn_op is original
    assert "forward" not in vars(original)
    assert original(None, None, None) == "original A2A"


@pytest.mark.parametrize("world_size", [1, 2])
def test_minimal_a2a_checks_context_parallel_world_size(
    a2a_definition, monkeypatch, world_size
):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            distributed=SimpleNamespace(get_world_size=lambda group: world_size)
        ),
    )
    original = MinimalA2AAttnOp(pg=object())
    attention = Attention(backend="minimal_a2a", attn_op=original)
    if world_size == 1:
        with module.cosmos_attention_backend(
            Network(attention), "sdpa_flash"
        ) as report:
            assert "forward" in vars(original)
            assert (
                report["coverage"]["blocks.0.self_attn"]["context_parallel_group"]
                == "singleton"
            )
    else:
        with pytest.raises(ValueError, match="context parallel"):
            with module.cosmos_attention_backend(Network(attention), "sdpa_flash"):
                pass
    assert "forward" not in vars(original)


@pytest.mark.parametrize(
    "kwargs",
    [{"local_attn": lambda *args: None}, {"training": True}],
)
def test_minimal_a2a_rejects_unknown_local_operation_or_training(
    a2a_definition, monkeypatch, kwargs
):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    original = MinimalA2AAttnOp(**kwargs)
    attention = Attention(backend="minimal_a2a", attn_op=original)
    with pytest.raises(ValueError, match="MinimalA2A"):
        with module.cosmos_attention_backend(Network(attention), "sdpa_flash"):
            pass
    assert "forward" not in vars(original)


def test_minimal_a2a_cannot_enable_context_parallel_inside_active_override(
    a2a_definition, monkeypatch
):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(distributed=SimpleNamespace(get_world_size=lambda group: 2)),
    )
    original = MinimalA2AAttnOp()
    attention = Attention(backend="minimal_a2a", attn_op=original)
    with module.cosmos_attention_backend(Network(attention), "sdpa_flash"):
        original.pg = object()
        with pytest.raises(ValueError, match="context parallel"):
            original(None, None, None)
    assert "forward" not in vars(original)


class DotProductAttention:
    __module__ = "transformer_engine.pytorch.attention"

    def __init__(self, **kwargs):
        self.training = False
        self.attn_mask_type = "no_mask"
        self.qkv_format = "bshd"
        self.attention_dropout = 0.0
        self.cp_group = None
        self.tp_size = 1
        self.window_size = (-1, -1)
        self.num_gqa_groups = 2
        self.flash_attention = SimpleNamespace(softmax_scale=0.25)
        vars(self).update(kwargs)

    def forward(self, q, k, v):
        return "original module"

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def test_transformer_engine_module_keeps_identity_scale_and_restores_forward(
    owned_attention, monkeypatch
):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    monkeypatch.setattr(
        module.AttentionOperator,
        "__call__",
        lambda self, q, k, v, **kwargs: kwargs,
    )
    original = DotProductAttention()
    attention = Attention(backend="transformer_engine", attn_op=original)
    with module.cosmos_attention_backend(Network(attention), "sdpa_flash") as report:
        assert attention.attn_op is original
        assert original(None, None, None) == {"scale": 0.25}
        assert report["coverage"]["blocks.0.self_attn"]["scale"] == 0.25
    assert attention.attn_op is original
    assert "forward" not in vars(original)
    assert original(None, None, None) == "original module"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attn_mask_type": "causal"},
        {"qkv_format": "sbhd"},
        {"window_size": (8, 8)},
        {"cp_group": object()},
        {"tp_size": 2},
        {"num_gqa_groups": 1},
        {"attention_dropout": 0.1},
    ],
)
def test_transformer_engine_nondefault_semantics_are_rejected(
    owned_attention, monkeypatch, kwargs
):
    module = api()
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    original = DotProductAttention(**kwargs)
    attention = Attention(backend="transformer_engine", attn_op=original)
    with pytest.raises(ValueError, match="Transformer Engine"):
        with module.cosmos_attention_backend(Network(attention), "sdpa_flash"):
            pass
    assert "forward" not in vars(original)


@pytest.fixture
def torch_cpu():
    return pytest.importorskip("torch", reason="Optional CPU numerical attention test")


@pytest.mark.parametrize("mask_kind", ["none", "boolean", "additive"])
def test_math_backend_preserves_mask_scale_cross_attention_and_output_layout(
    torch_cpu, mask_kind
):
    torch = torch_cpu
    module = api()
    torch.manual_seed(7)
    q = torch.randn(2, 3, 2, 8)
    k = torch.randn(2, 5, 2, 8)
    v = torch.randn_like(k)
    mask = None
    if mask_kind == "boolean":
        mask = torch.ones(3, 5, dtype=torch.bool)
        mask[:, -1] = False
    elif mask_kind == "additive":
        mask = torch.randn(3, 5) * 0.1
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        scale=0.25,
    ).transpose(1, 2)
    record = {}
    operator = module.AttentionOperator("sdpa_math", record=record)
    actual = operator(q, k, v, attn_mask=mask, scale=0.25, flatten_heads=False)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        operator(q, k, v, attn_mask=mask, scale=0.25), expected.flatten(2)
    )
    assert record["python_calls"] == 2
    assert record["observed_shapes"][0]["query"] == [2, 3, 2, 8]
    assert record["observed_shapes"][0]["key"] == [2, 5, 2, 8]


def test_math_backend_preserves_causal_and_dropout_arguments(torch_cpu):
    torch = torch_cpu
    module = api()
    q = torch.randn(1, 3, 2, 8)
    operator = module.AttentionOperator("sdpa_math")
    torch.manual_seed(5)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                q.transpose(1, 2),
                q.transpose(1, 2),
                dropout_p=0.2,
                is_causal=True,
            )
            .transpose(1, 2)
            .flatten(2)
        )
    torch.manual_seed(5)
    actual = operator(q, q, q, dropout_p=0.2, is_causal=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["sdpa_flash", "sdpa_efficient", "flash_attn"])
def test_accelerated_backends_do_not_fallback_to_cpu(torch_cpu, backend, monkeypatch):
    module = api()
    # Rejection is independent of whether the optional flash-attn wheel is installed.
    monkeypatch.setattr(module, "_load_backend", lambda backend: object())
    q = torch_cpu.randn(1, 3, 2, 8)
    with pytest.raises(ValueError, match="CUDA"):
        module.AttentionOperator(backend)(q, q, q)
