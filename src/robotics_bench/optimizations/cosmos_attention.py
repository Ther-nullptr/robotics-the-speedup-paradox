"""Explicit dense-attention alternatives for the owned Cosmos DiT.

The model adapter changes only the attention operation. Q/K normalization, RoPE,
projections, and output dropout stay in the model. Backend selection never falls
back: an unsupported device, mask, or kernel raises. This is an inference-only,
single-thread adapter, not a replacement for sparse or context-parallel attention.
"""

from contextlib import contextmanager
from importlib import import_module
from math import isfinite
import sys

from .patching import Patches


BACKENDS = ("sdpa_flash", "sdpa_efficient", "sdpa_math", "flash_attn")


def _validate_backend(backend):
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown attention backend {backend!r}; choose from {BACKENDS}"
        )


def _load_backend(backend):
    """Load one exact implementation without initializing CUDA or importing Cosmos."""
    if backend == "flash_attn":
        try:
            from flash_attn import flash_attn_func
        except (ImportError, OSError) as error:
            raise ImportError(
                "flash_attn requires a compatible installed flash-attn package"
            ) from error

        def run(q, k, v, mask, dropout, causal, scale):
            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout,
                softmax_scale=scale,
                causal=causal,
                return_attn_probs=False,
            )

        return run

    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    selected = {
        "sdpa_flash": SDPBackend.FLASH_ATTENTION,
        "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "sdpa_math": SDPBackend.MATH,
    }[backend]

    def run(q, k, v, mask, dropout, causal, scale):
        # A singleton backend list disables math/cuDNN/other fused fallbacks.
        # Keep the context local: VAE and unrelated models retain their settings.
        with sdpa_kernel(selected):
            result = scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask,
                dropout_p=dropout,
                is_causal=causal,
                scale=scale,
            )
        return result.transpose(1, 2)

    return run


class AttentionOperator:
    """B,S,H,D attention with the owned Cosmos output and SDPA mask conventions.

    Boolean masks use True for allowed positions; floating masks add to the logits.
    Default scaling is 1/sqrt(head_dim). Causal masks follow SDPA's top-left alignment.
    FlashAttention's non-square causal mode has different alignment and is rejected.
    Call counts are Python invocations, not CUDA graph replay counts.
    """

    def __init__(self, backend, *, record=None):
        _validate_backend(backend)
        self.backend = backend
        self._run = _load_backend(backend)
        self.record = record if record is not None else {}
        self.record.update(
            backend=backend,
            fallback=False,
            python_calls=0,
            observed_shapes=[],
        )
        self._seen_shapes = set()

    def __call__(
        self,
        q,
        k,
        v,
        attn_mask=None,
        flatten_heads=True,
        *,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        video_size=None,
    ):
        # video_size is metadata from the original dense i4 adapter; it does not
        # alter that operation's full, bidirectional attention semantics.
        if any(x.ndim != 4 for x in (q, k, v)):
            raise ValueError("Attention inputs must use B,S,H,D layout")
        if (
            q.shape[0] != k.shape[0]
            or k.shape != v.shape
            or q.shape[2:] != k.shape[2:]
            or any(dimension <= 0 for x in (q, k, v) for dimension in x.shape)
        ):
            raise ValueError(
                "Dense Cosmos attention requires matching batch, heads, and head_dim"
            )
        if q.device != k.device or k.device != v.device:
            raise ValueError("Attention inputs must share a device")
        if q.dtype != k.dtype or k.dtype != v.dtype:
            raise ValueError("Attention inputs must share a dtype")
        if self.backend != "sdpa_math" and q.device.type != "cuda":
            raise ValueError(
                f"{self.backend} requires CUDA inputs; fallback is disabled"
            )
        if attn_mask is not None and is_causal:
            raise ValueError("Use either an explicit attention mask or is_causal")
        if not 0 <= dropout_p <= 1:
            raise ValueError("dropout_p must be between zero and one")
        if self.backend == "flash_attn":
            if attn_mask is not None:
                raise ValueError(
                    "flash_attn does not support an explicit attention mask"
                )
            if is_causal and q.shape[1] != k.shape[1]:
                raise ValueError(
                    "flash_attn non-square causal alignment differs from SDPA"
                )
        result = self._run(q, k, v, attn_mask, dropout_p, is_causal, scale)
        self.record["python_calls"] += 1
        shape = (tuple(q.shape), tuple(k.shape), str(q.dtype), str(q.device))
        if shape not in self._seen_shapes:
            self._seen_shapes.add(shape)
            self.record["observed_shapes"].append(
                {
                    "query": list(q.shape),
                    "key": list(k.shape),
                    "value": list(v.shape),
                    "dtype": str(q.dtype),
                    "device": str(q.device),
                }
            )
        return result.flatten(2) if flatten_heads else result


def _transformer_engine_scale(module):
    """Accept only the no-mask, unsharded DPA constructed by owned Attention."""
    original = module.attn_op
    if (
        type(original).__name__ != "DotProductAttention"
        or not type(original).__module__.startswith("transformer_engine.pytorch.")
        or original.training
        or getattr(original, "qkv_format", None) != "bshd"
        or getattr(original, "attn_mask_type", None) != "no_mask"
        or getattr(original, "attention_dropout", None) != 0.0
        or getattr(original, "window_size", None) != (-1, -1)
        or getattr(original, "cp_group", None) is not None
        or getattr(original, "tp_size", None) != 1
        or getattr(original, "num_gqa_groups", None) != module.n_heads
    ):
        raise ValueError("Unsupported Transformer Engine attention configuration")
    scale = getattr(getattr(original, "flash_attention", None), "softmax_scale", None)
    if not isinstance(scale, (int, float)) or not isfinite(scale):
        raise ValueError("Unsupported Transformer Engine attention scale")
    return scale


def _module_forward(operator, scale):
    def forward(module, q, k, v):
        # Cosmos invokes DPA with these three positional tensors. Extra DPA
        # features (cache, masks, packed sequences) must fail, not be discarded.
        return operator(q, k, v, scale=scale)

    return forward


def _minimal_a2a_world_size(original):
    if not hasattr(original, "pg"):
        raise ValueError("Unsupported MinimalA2A context parallel state")
    if original.pg is None:
        return 1
    from torch import distributed

    try:
        world_size = distributed.get_world_size(original.pg)
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValueError("Cannot validate MinimalA2A context parallel group") from error
    if world_size != 1:
        raise ValueError("MinimalA2A context parallel world size must equal one")
    return world_size


def _validate_minimal_a2a(module):
    original = module.attn_op
    definition = sys.modules.get("cosmos_policy._src.predict2.networks.a2a_cp")
    if (
        definition is None
        or type(original) is not getattr(definition, "MinimalA2AAttnOp", None)
        or original.training
        or getattr(original, "local_attn", None)
        is not getattr(definition, "attention", None)
    ):
        # Exact type excludes sparse Natten subclasses and AttentionOpWithKVCache.
        raise ValueError("Unsupported MinimalA2A attention operator or configuration")
    return _minimal_a2a_world_size(original)


def _minimal_a2a_forward(operator):
    def forward(module, query, key, value):
        # Context parallelism can be changed after entering this override.
        # Never bypass a newly enabled all-to-all communication path.
        _minimal_a2a_world_size(module)
        import torch

        # predict2.networks.attention defaults to BF16 independently of input
        # dtype. Its output is contiguous B,S,H,D, then MinimalA2A flattens heads.
        return operator(
            query.to(torch.bfloat16),
            key.to(torch.bfloat16),
            value.to(torch.bfloat16),
        ).contiguous()

    return forward


@contextmanager
def cosmos_attention_backend(network, backend):
    """Temporarily replace all supported attention operators on ``model.net``.

    Example::

        with cosmos_attention_backend(model.net, "sdpa_flash") as report:
            actions = infer(observation)

    The yielded JSON-serializable report records selected backend, coverage and
    shapes actually observed. Full-model numerical checks and profiler traces are
    still required; changing attention algorithms can change floating-point sums.
    """
    _validate_backend(backend)
    dit = import_module("robotics_bench.models.cosmos.minimal_v4_dit")
    modules = [
        (name, module)
        for name, module in network.named_modules()
        if isinstance(module, dit.Attention)
    ]
    if not modules:
        raise ValueError("No owned Cosmos attention modules matched")
    scales = {}
    context_sizes = {}
    for name, module in modules:
        if module.training:
            raise ValueError("Cosmos attention replacement is inference-only")
        if module.qkv_format != "bshd":
            raise ValueError(f"{name}: only bshd attention layout is supported")
        if module.backend == "transformer_engine":
            scales[name] = _transformer_engine_scale(module)
            continue
        if module.backend == "minimal_a2a":
            context_sizes[name] = _validate_minimal_a2a(module)
            continue
        if module.backend not in ("i4", "torch", "torch-flex"):
            raise ValueError(
                f"{name}: Unsupported original attention backend {module.backend!r}"
            )
        operation_name = {
            "i4": "i4_attention_op",
            "torch": "torch_attention_op",
            "torch-flex": "flex_attention_op",
        }[module.backend]
        # The native loader constructs first, then bind_owned_runtime switches
        # classes. Existing attn_op attributes can retain the upstream function.
        upstream = sys.modules.get(
            "cosmos_policy._src.predict2.networks.minimal_v4_dit"
        )
        known_operations = (
            getattr(dit, operation_name),
            getattr(upstream, operation_name, None),
        )
        if not any(module.attn_op is operation for operation in known_operations):
            raise ValueError(
                f"{name}: Unsupported attention operator; refusing to change its semantics"
            )
    report = {
        "backend": backend,
        "adapter": __name__,
        "fallback": False,
        "call_count_scope": "Python operator calls; excludes CUDA graph replays",
        "coverage": {},
    }
    patches = Patches()
    try:
        for name, module in modules:
            record = {
                "original_backend": module.backend,
                "kind": "self" if module.is_selfattn else "cross",
                "head_dim": module.head_dim,
                "heads": module.n_heads,
                "scale": scales.get(name),
            }
            operator = AttentionOperator(backend, record=record)
            if module.backend == "transformer_engine":
                # Preserve the registered nn.Module and its state; only bind an
                # instance forward, so restoration also keeps module ownership.
                patches.bind(
                    module.attn_op, "forward", _module_forward(operator, scales[name])
                )
            elif module.backend == "minimal_a2a":
                record["context_parallel_size"] = context_sizes[name]
                record["context_parallel_group"] = (
                    None if module.attn_op.pg is None else "singleton"
                )
                record["input_conversion"] = "bfloat16"
                patches.bind(module.attn_op, "forward", _minimal_a2a_forward(operator))
            else:
                patches.set(module, "attn_op", operator)
            report["coverage"][name] = record
        yield report
    finally:
        patches.restore()
