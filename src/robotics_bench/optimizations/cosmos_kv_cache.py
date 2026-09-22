"""Request-scoped native cross-attention K/V, refreshed before graph execution.

Install after quantization and call ``cache.prepare(raw_context)`` outside the
DiT CUDA graph before every forward. Include ``layout_generation`` in graph
invalidation and supply ``retained_tensors`` to the graph owner. The cache never
reuses projected values across requests, including repeated prompts.
"""

from contextlib import contextmanager, nullcontext
from importlib import import_module

from .patching import Patches


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Tensors created in torch.inference_mode have no version counter.
        return None


def _signature(tensor):
    return tuple(tensor.shape), tensor.dtype, tensor.device, tuple(tensor.stride())


def _configuration(modules):
    result = []
    for name, attention in modules:
        for attr in ("k_proj", "v_proj", "k_norm", "v_norm"):
            module = getattr(attention, attr)
            owner = module
            group = getattr(module, "_group", None)
            if callable(group):
                owner = group().linear
            tensors = tuple(owner.parameters()) + tuple(owner.buffers())
            result.append(
                (
                    name,
                    attr,
                    id(module),
                    getattr(owner, "backend", None),
                    getattr(owner, "bits", None),
                    getattr(owner, "tactic", None),
                    tuple((id(t), _signature(t), _version(t)) for t in tensors),
                )
            )
    return tuple(result)


class CrossAttentionCache:
    """Own one fixed-layout cache and serialized request lifecycle.

    Versioned raw tensors are checked for mutation using their counter. Inference
    tensors use an exact comparison against a private snapshot; that comparison
    can synchronize CUDA and is intentionally included in service measurements.
    Weight changes require reinstalling this context, as with the enclosing graph.
    """

    def __init__(self, model, modules):
        self.model = model
        self.modules = tuple(modules)
        self.layout_generation = 0
        self.request_generation = 0
        self._configuration = _configuration(self.modules)
        self._buffers = {}
        self._layout = None
        self._active = self._prepared = self._closed = False
        self._raw = self._snapshot = self._verified_clone = None
        self._raw_signature = self._raw_version = None
        self.report = {
            "adapter": __name__,
            "scope": "one action request; native cross K/V projections and norms",
            "requests_started": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "prepare_calls": 0,
            "refreshes": 0,
            "layout_generation": 0,
            "request_generation": 0,
            "buffer_bytes": 0,
            "mutation_checks": {"version_counter": 0, "exact_tensor_compare": 0},
            "coverage": {name: {"refreshes": 0} for name, _ in self.modules},
        }

    def _require_request(self):
        if self._closed:
            raise RuntimeError("Cross-attention cache is closed")
        if not self._active:
            raise RuntimeError("Cross-attention cache needs an active request")

    def _guard_environment(self):
        import torch

        if self.model.net.training or any(m.training for _, m in self.modules):
            raise ValueError("Cross-attention caching does not support training")
        if torch.is_grad_enabled():
            raise ValueError("Cross-attention caching requires no-grad inference")
        if getattr(self.model.net, "is_context_parallel_enabled", False):
            raise ValueError(
                "Cross-attention caching does not support context parallel"
            )
        for _, module in self.modules:
            for attr in ("pg", "cp_group", "_cp_group"):
                if getattr(module.attn_op, attr, None) is not None:
                    raise ValueError(
                        "Cross-attention caching does not support context parallel"
                    )

    @contextmanager
    def request(self):
        if self._closed:
            raise RuntimeError("Cross-attention cache is closed")
        if self._active:
            raise RuntimeError("Cross-attention cache requests must be serialized")
        self._active = True
        self._prepared = False
        self.request_generation += 1
        self.report["request_generation"] = self.request_generation
        self.report["requests_started"] += 1
        try:
            yield
            if not self._prepared:
                raise RuntimeError(
                    "Action request never called cross-attention prepare"
                )
            self.report["requests_completed"] += 1
        except BaseException:
            self.report["requests_failed"] += 1
            raise
        finally:
            self._active = self._prepared = False
            self._raw = self._snapshot = self._verified_clone = None
            self._raw_signature = self._raw_version = None

    def prepare(self, context):
        """Validate raw context every call, refresh native K/V once per request."""
        import torch

        self._require_request()
        self._guard_environment()
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("Cross-attention cache requires one B,S,D text tensor")
        if not context.is_floating_point() or min(context.shape) <= 0:
            raise ValueError("Invalid cross-attention context")
        if context.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Cross-attention prepare must run before CUDA graph capture"
            )
        self.report["prepare_calls"] += 1
        if self._prepared:
            if context is not self._raw:
                raise ValueError(
                    "One request must retain the same context object; CFG is unsupported"
                )
            if _signature(context) != self._raw_signature:
                raise ValueError("Raw context was modified during the request")
            if self._raw_version is not None:
                self.report["mutation_checks"]["version_counter"] += 1
                unchanged = _version(context) == self._raw_version
            else:
                self.report["mutation_checks"]["exact_tensor_compare"] += 1
                unchanged = torch.equal(context, self._snapshot)
            if not unchanged:
                raise ValueError("Raw context was modified during the request")
            return
        if _configuration(self.modules) != self._configuration:
            raise RuntimeError(
                "Projection weights or precision changed; reinstall the cache"
            )
        # A unique snapshot also prevents stale sibling-pack reuse after a failed
        # request whose native shared-quant cache stopped between K and V.
        snapshot = context.clone()
        computed = {}
        for name, module in self.modules:
            if context.shape[-1] != module.context_dim:
                raise ValueError("Text context width does not match cross-attention")
            shape = (*context.shape[:-1], module.n_heads, module.head_dim)
            key = module.k_proj(snapshot).reshape(shape)
            value = module.v_proj(snapshot).reshape(shape)
            computed[name] = (module.k_norm(key), module.v_norm(value))
        layout = tuple(
            (name, tuple(_signature(t) for t in pair))
            for name, pair in computed.items()
        )
        buffers = self._buffers
        if layout != self._layout:
            buffers = {
                name: tuple(
                    torch.empty_strided(
                        t.shape, t.stride(), dtype=t.dtype, device=t.device
                    )
                    for t in pair
                )
                for name, pair in computed.items()
            }
        # Do not overwrite any previous buffer until every projection succeeded.
        for name, pair in computed.items():
            for target, value in zip(buffers[name], pair):
                target.copy_(value)
        if layout != self._layout:
            self.layout_generation += 1
        self._buffers, self._layout = buffers, layout
        self._raw, self._snapshot = context, snapshot
        self._raw_signature, self._raw_version = _signature(context), _version(context)
        self._prepared = True
        self.report["refreshes"] += 1
        self.report["layout_generation"] = self.layout_generation
        self.report["buffer_bytes"] = sum(
            t.untyped_storage().nbytes() for t in self.retained_tensors()
        )
        for name, pair in buffers.items():
            self.report["coverage"][name].update(
                refreshes=self.report["coverage"][name]["refreshes"] + 1,
                key_shape=list(pair[0].shape),
                value_shape=list(pair[1].shape),
                dtype=str(pair[0].dtype),
            )

    def compute_qkv(self, name, module, x, context):
        import torch

        self._require_request()
        if not self._prepared:
            raise RuntimeError("Call cross-attention prepare before DiT execution")
        if torch.is_grad_enabled() or module.training:
            raise ValueError("Cached cross attention requires no-grad inference")
        if (
            not isinstance(context, torch.Tensor)
            or _signature(context) != self._raw_signature
        ):
            raise ValueError("Unexpected or unverified cross-attention context")
        if context is not self._raw and context is not self._verified_clone:
            if context.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Unverified graph context must be validated during warmup"
                )
            if not torch.equal(context, self._snapshot):
                raise ValueError("Unexpected or unverified cross-attention context")
            # CUDA graph warmup clones the validated raw input. Registration
            # occurs outside capture; later raw-input validation stays in prepare.
            self._verified_clone = context
        q = module.q_proj(x)
        q = module.q_norm(q.reshape(*q.shape[:-1], module.n_heads, module.head_dim))
        key, value = self._buffers[name]
        return q, key, value

    def retained_tensors(self):
        return tuple(tensor for pair in self._buffers.values() for tensor in pair)

    def close(self):
        self._closed = True
        self._buffers = {}
        self._raw = self._snapshot = self._verified_clone = None


@contextmanager
def cache_cross_attention(model):
    """Patch plain text cross attention and compose the engine's request scope."""
    import torch

    dit = import_module("robotics_bench.models.cosmos.minimal_v4_dit")
    net = model.net
    if net.training:
        raise ValueError("Cross-attention caching does not support training")
    if getattr(net, "is_context_parallel_enabled", False):
        raise ValueError("Cross-attention caching does not support context parallel")
    if getattr(net, "extra_image_context_dim", None) is not None:
        raise ValueError("Image cross attention is unsupported")
    if getattr(net, "use_crossattn_projection", False) and not isinstance(
        getattr(net, "crossattn_proj", None), torch.nn.Identity
    ):
        raise ValueError("Nonidentity crossattn_proj is unsupported")
    modules = []
    for name, module in net.named_modules():
        if not isinstance(module, dit.Attention) or module.is_selfattn:
            continue
        if (
            type(module) is not dit.Attention
            or module.qkv_format != "bshd"
            or module.training
        ):
            raise ValueError("Only plain inference text cross Attention is supported")
        if not isinstance(module.v_norm, torch.nn.Identity):
            raise ValueError("Unsupported cross-attention value normalization")
        if any(
            getattr(module.attn_op, attr, None) is not None
            for attr in ("pg", "cp_group", "_cp_group")
        ):
            raise ValueError(
                "Cross-attention caching does not support context parallel"
            )
        modules.append((name, module))
    if not modules:
        raise ValueError("No plain text cross Attention modules matched")
    cache = CrossAttentionCache(model, modules)
    previous = getattr(model, "_robotics_action_only_context", None)
    if previous is not None and not callable(previous):
        raise ValueError("Existing action request context must be callable")
    patches = Patches()

    @contextmanager
    def action_request(**kwargs):
        with cache.request():
            with previous(**kwargs) if previous is not None else nullcontext():
                yield

    def bind(name):
        def compute_qkv(module, x, context=None, rope_emb=None):
            # Native cross attention never applies RoPE; Q remains dynamic.
            return cache.compute_qkv(name, module, x, context)

        return compute_qkv

    try:
        patches.set(model, "_robotics_action_only_context", action_request)
        for name, module in modules:
            patches.bind(module, "compute_qkv", bind(name))
        yield cache
    finally:
        patches.restore()
        cache.close()
