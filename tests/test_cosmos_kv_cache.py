"""Request-scoped text K/V reuse keeps native math and rejects stale contexts."""

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def api():
    path = (
        Path(__file__).parents[1]
        / "src/robotics_bench/optimizations/cosmos_kv_cache.py"
    )
    assert path.exists(), "Cosmos cross-attention cache is missing"
    return importlib.import_module("robotics_bench.optimizations.cosmos_kv_cache")


class CountingLinear(torch.nn.Linear):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs, bias=False)
        self.calls = 0
        self.fail = False

    def forward(self, x):
        self.calls += 1
        if self.fail:
            raise RuntimeError("projection failed")
        return super().forward(x)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_selfattn = False
        self.qkv_format = "bshd"
        self.n_heads = 2
        self.head_dim = 3
        self.context_dim = 4
        self.q_proj = CountingLinear(6, 6)
        self.k_proj = CountingLinear(4, 6)
        self.v_proj = CountingLinear(4, 6)
        self.q_norm = torch.nn.RMSNorm(3, eps=1e-6)
        self.k_norm = torch.nn.RMSNorm(3, eps=1e-6)
        self.v_norm = torch.nn.Identity()
        self.attn_op = SimpleNamespace(pg=None)

    def compute_qkv(self, x, context=None, rope_emb=None):
        q = self.q_proj(x).reshape(*x.shape[:-1], self.n_heads, self.head_dim)
        k = self.k_proj(context).reshape(
            *context.shape[:-1], self.n_heads, self.head_dim
        )
        v = self.v_proj(context).reshape(
            *context.shape[:-1], self.n_heads, self.head_dim
        )
        return self.q_norm(q), self.k_norm(k), self.v_norm(v)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "robotics_bench.models.cosmos.minimal_v4_dit",
        SimpleNamespace(Attention=Attention),
    )
    model = torch.nn.Module()
    model.net = torch.nn.Module()
    model.net.blocks = torch.nn.ModuleList([Attention(), Attention()])
    model.net.use_crossattn_projection = False
    model.net.extra_image_context_dim = None
    model.net.is_context_parallel_enabled = False
    return model.eval()


@torch.no_grad()
def test_a_b_a_requests_refresh_once_and_reuse_fixed_addresses(model):
    module = api()
    x = torch.randn(1, 5, 6)
    a, b = torch.randn(1, 7, 4), torch.randn(1, 7, 4)
    expected = [
        [layer.compute_qkv(x, c) for layer in model.net.blocks] for c in (a, b, a)
    ]
    for layer in model.net.blocks:
        layer.k_proj.calls = layer.v_proj.calls = 0
    with module.cache_cross_attention(model) as cache:
        addresses = None
        for context, reference in zip((a, b, a), expected):
            with model._robotics_action_only_context(suite="test"):
                for _ in range(5):
                    cache.prepare(context)
                    current = tuple(t.data_ptr() for t in cache.retained_tensors())
                    addresses = current if addresses is None else addresses
                    assert current == addresses
                    for layer, ref in zip(model.net.blocks, reference):
                        actual = layer.compute_qkv(x, context)
                        assert all(torch.equal(v, r) for v, r in zip(actual, ref))
        assert cache.layout_generation == 1
        assert cache.report["refreshes"] == 3
        assert cache.report["prepare_calls"] == 15
        assert cache.report["requests_completed"] == 3
        assert cache.request_generation == cache.report["request_generation"] == 3
        assert all(
            layer.k_proj.calls == layer.v_proj.calls == 3 for layer in model.net.blocks
        )
    assert "_robotics_action_only_context" not in vars(model)
    assert all("compute_qkv" not in vars(layer) for layer in model.net.blocks)


@pytest.mark.parametrize("inference_tensor", [False, True])
@torch.no_grad()
def test_rejects_context_replacement_and_mutation_in_one_request(
    model, inference_tensor
):
    module = api()
    with torch.inference_mode(inference_tensor), torch.no_grad():
        context = torch.randn(1, 7, 4)
        with module.cache_cross_attention(model) as cache:
            with model._robotics_action_only_context():
                cache.prepare(context)
                with pytest.raises(ValueError, match="same context"):
                    cache.prepare(context.clone())
                context.add_(1)
                with pytest.raises(ValueError, match="modified"):
                    cache.prepare(context)


@torch.no_grad()
def test_lifecycle_composes_previous_context_and_recovers_after_error(model):
    module = api()
    events = []

    @contextmanager
    def previous(**kwargs):
        events.append(("enter", kwargs))
        try:
            yield
        finally:
            events.append(("exit", kwargs))

    model._robotics_action_only_context = previous
    context = torch.randn(1, 7, 4)
    with module.cache_cross_attention(model) as cache:
        with pytest.raises(RuntimeError, match="active request"):
            cache.prepare(context)
        with pytest.raises(RuntimeError, match="request failed"):
            with model._robotics_action_only_context(tag=1):
                cache.prepare(context)
                with pytest.raises(RuntimeError, match="serialized"):
                    with model._robotics_action_only_context():
                        pass
                raise RuntimeError("request failed")
        with model._robotics_action_only_context(tag=2):
            cache.prepare(context)
        assert cache.report["requests_failed"] == 1
        assert cache.report["requests_completed"] == 1
    assert model._robotics_action_only_context is previous
    assert ("enter", {"tag": 1}) in events and ("exit", {"tag": 1}) in events
    assert ("enter", {"tag": 2}) in events and ("exit", {"tag": 2}) in events
    with pytest.raises(RuntimeError, match="closed"):
        cache.prepare(context)


@torch.no_grad()
def test_previous_factory_exit_validation_counts_as_failed_request(model):
    module = api()

    @contextmanager
    def previous(**kwargs):
        yield
        raise RuntimeError("prefix exit validation failed")

    model._robotics_action_only_context = previous
    context = torch.randn(1, 7, 4)
    with module.cache_cross_attention(model) as cache:
        with pytest.raises(RuntimeError, match="prefix exit validation failed"):
            with model._robotics_action_only_context():
                cache.prepare(context)
        assert cache.report["requests_started"] == 1
        assert cache.report["requests_completed"] == 0
        assert cache.report["requests_failed"] == 1
        with pytest.raises(RuntimeError, match="active request"):
            cache.prepare(context)
    assert model._robotics_action_only_context is previous


@torch.no_grad()
def test_shape_change_advances_layout_generation_but_failed_refresh_is_transactional(
    model,
):
    module = api()
    context = torch.randn(1, 7, 4)
    with module.cache_cross_attention(model) as cache:
        with model._robotics_action_only_context():
            cache.prepare(context)
        original = cache.retained_tensors()
        model.net.blocks[-1].k_proj.fail = True
        with pytest.raises(RuntimeError, match="projection failed"):
            with model._robotics_action_only_context():
                cache.prepare(torch.randn(1, 9, 4))
        assert all(a is b for a, b in zip(cache.retained_tensors(), original))
        assert cache.layout_generation == 1
        model.net.blocks[-1].k_proj.fail = False
        with model._robotics_action_only_context():
            cache.prepare(torch.randn(1, 9, 4))
        assert cache.layout_generation == 2
        assert all(t.shape == (1, 9, 2, 3) for t in cache.retained_tensors())


@torch.no_grad()
def test_unprepared_compute_and_unverified_context_clone_are_rejected(model):
    module = api()
    x = torch.randn(1, 5, 6)
    context = torch.randn(1, 7, 4)
    with module.cache_cross_attention(model) as cache:
        with model._robotics_action_only_context():
            with pytest.raises(RuntimeError, match="prepare"):
                model.net.blocks[0].compute_qkv(x, context)
            cache.prepare(context)
            clone = context.clone()
            model.net.blocks[0].compute_qkv(x, clone)
            with pytest.raises(ValueError, match="unverified"):
                model.net.blocks[0].compute_qkv(x, torch.randn_like(context))


@torch.no_grad()
def test_compute_cannot_reenable_autograd_after_prepare(model):
    module = api()
    x, context = torch.randn(1, 5, 6), torch.randn(1, 7, 4)
    with module.cache_cross_attention(model) as cache:
        with model._robotics_action_only_context():
            cache.prepare(context)
            with torch.enable_grad():
                with pytest.raises(ValueError, match="no-grad"):
                    model.net.blocks[0].compute_qkv(x, context)


@pytest.mark.parametrize(
    "change", ["projection", "weights", "training", "context_parallel"]
)
@torch.no_grad()
def test_configuration_changes_require_reinstall(model, change):
    module = api()
    context = torch.randn(1, 7, 4)
    with module.cache_cross_attention(model) as cache:
        with model._robotics_action_only_context():
            cache.prepare(context)
        if change == "projection":
            model.net.blocks[0].k_proj = CountingLinear(4, 6)
        elif change == "weights":
            model.net.blocks[0].k_proj.weight.add_(1)
        elif change == "training":
            model.net.train()
        else:
            model.net.is_context_parallel_enabled = True
        with pytest.raises(
            (ValueError, RuntimeError), match="reinstall|training|parallel"
        ):
            with model._robotics_action_only_context():
                cache.prepare(context)


@pytest.mark.parametrize(
    "unsupported", ["image_cross", "projection", "parallel", "training"]
)
def test_rejects_unsupported_model_without_leaking_patches(model, unsupported):
    module = api()
    if unsupported == "image_cross":
        model.net.extra_image_context_dim = 4
    elif unsupported == "projection":
        model.net.use_crossattn_projection = True
        model.net.crossattn_proj = torch.nn.Linear(4, 4)
    elif unsupported == "parallel":
        model.net.is_context_parallel_enabled = True
    else:
        model.net.train()
    with pytest.raises(ValueError):
        with module.cache_cross_attention(model):
            pass
    assert "_robotics_action_only_context" not in vars(model)
    assert all("compute_qkv" not in vars(layer) for layer in model.net.blocks)
