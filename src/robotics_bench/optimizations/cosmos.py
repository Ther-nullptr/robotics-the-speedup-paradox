"""Independent Cosmos DiT switches with native normalization preserved."""

from contextlib import contextmanager, ExitStack
from .patching import Patches
from .config import ATTENTION_SWITCHES, CONVOLUTION_SWITCHES, COSMOS_HOTSPOT_SWITCHES


def _modulation(x, norm, scale, shift, enabled):
    y = norm(x)
    formats = getattr(norm, "_robotics_quant_bits", ())
    if formats:
        from robotics_kernels.ampere_ada.modulation import (
            PackedActivations,
            prepare_modulation,
        )

        b, t, h, w, d = x.shape
        scale = scale.reshape(b * t, 1, d).contiguous()
        shift = shift.reshape(b * t, 1, d).contiguous()
        packs = {
            bits: prepare_modulation(y.reshape(b * t, h * w, d), scale, shift, bits)
            for bits in formats
        }
        return PackedActivations(packs).with_leading_shape(x.shape[:-1])
    if not enabled:
        return y * (1 + scale) + shift
    from robotics_kernels.fused import modulate

    b, t, h, w, d = x.shape
    return modulate(
        y.reshape(b * t, h * w, d),
        scale.reshape(b * t, 1, d).contiguous(),
        shift.reshape(b * t, 1, d).contiguous(),
    ).reshape_as(x)


def _attention_input(value, leading_shape):
    if getattr(value, "_robotics_packed_input", False):
        return value.with_leading_shape(leading_shape)
    return value.reshape(*leading_shape, value.shape[-1])


def _residual(x, y, gate, enabled):
    if not enabled:
        return x + gate * y
    from robotics_kernels.fused import gated_residual

    b, t, h, w, d = x.shape
    return gated_residual(
        x.reshape(b * t, h * w, d),
        y.reshape(b * t, h * w, d).contiguous(),
        gate.reshape(b * t, 1, d).contiguous(),
    ).reshape_as(x)


@contextmanager
def optimize_cosmos(model, config):
    from robotics_bench.models.cosmos import minimal_v4_dit as dit

    if set(config.enabled) - {
        *COSMOS_HOTSPOT_SWITCHES,
        "modulation",
        "gated_residual",
        "cuda_graph",
        "shared_quant",
        "activation_quant_fusion",
        "integer_grouped",
        "integer_pack_reuse",
        "integer_qkv",
        "integer_group_views",
        "integer_biasless",
        "modulation_quant",
    }:
        raise ValueError("Unsupported Cosmos optimization switch")
    patches = Patches()
    stack = ExitStack()
    report = {
        "switches": list(config.enabled),
        "precision": config.precision,
        "coverage": {},
    }
    try:
        import torch

        report["cudnn"] = {
            name: getattr(torch.backends.cudnn, name, None)
            for name in (
                "enabled",
                "benchmark",
                "benchmark_limit",
                "deterministic",
                "allow_tf32",
            )
        }
        for switch, backend in ATTENTION_SWITCHES.items():
            if switch in config.enabled:
                from .cosmos_attention import cosmos_attention_backend

                report["attention"] = stack.enter_context(
                    cosmos_attention_backend(model.net, backend)
                )
        if "vae_norm_fusion" in config.enabled:
            from .cosmos_pointwise import optimize_vae_pointwise

            report["vae_pointwise"] = stack.enter_context(
                optimize_vae_pointwise(
                    model.tokenizer.model.model,
                    fuse_silu="vae_silu_fusion" in config.enabled,
                )
            )
        for switch, tactic in CONVOLUTION_SWITCHES.items():
            if switch in config.enabled:
                from .cosmos_convolution import optimize_convolutions

                report["convolution"] = stack.enter_context(
                    optimize_convolutions(
                        model.tokenizer.model.model,
                        tactic=tactic,
                        policy="c96"
                        if switch.startswith("conv_cutlass_c96")
                        else "all_supported",
                    )
                )
        if "modulation_quant" in config.enabled and (
            not config.precision.startswith("int") or "modulation" not in config.enabled
        ):
            raise ValueError(
                "modulation_quant requires integer precision and modulation"
            )
        for name, module in model.net.named_modules():
            if isinstance(module, dit.Block):
                patches.set(
                    module, "_robotics_modulation", "modulation" in config.enabled
                )
                patches.set(
                    module, "_robotics_residual", "gated_residual" in config.enabled
                )
                report["coverage"][name] = {
                    "modulation": "modulation" in config.enabled,
                    "residual": "gated_residual" in config.enabled,
                }
        if config.precision != "bf16":
            if config.scopes != ("dit",):
                raise ValueError("Cosmos quantization scope must be dit")
            from .quantization import quantize_linears

            def selector(name):
                return (
                    "dit"
                    if name.startswith("blocks.")
                    and (
                        ".self_attn." in name
                        or ".cross_attn." in name
                        or ".mlp." in name
                    )
                    else None
                )

            report["quantization"] = stack.enter_context(
                quantize_linears(
                    model.net,
                    config.precision,
                    scopes=config.scopes,
                    shared_quant="shared_quant" in config.enabled,
                    tactic=config.tactic,
                    selector=selector,
                    grouped="integer_grouped" in config.enabled,
                    pack_reuse="integer_pack_reuse" in config.enabled,
                    group_qkv="integer_qkv" in config.enabled,
                    group_views="integer_group_views" in config.enabled,
                    biasless_epilogue="integer_biasless" in config.enabled,
                    site_bits=_tier_sites(model.net, selector, config.quant_tier),
                )
            )
        if "activation_quant_fusion" in config.enabled:
            from robotics_kernels.integer import IntegerLinear

            if not config.precision.startswith("int"):
                raise ValueError("activation_quant_fusion requires an integer backend")
            for name, module in model.net.named_modules():
                if isinstance(module, dit.GPT2FeedForward) and isinstance(
                    module.layer2, IntegerLinear
                ):

                    def forward_mlp(module, x):
                        return module.layer2.forward_gelu(
                            module.layer1(x).contiguous(), approximate="none"
                        )

                    patches.bind(module, "forward", forward_mlp)
                    report["coverage"][name + ".activation"] = "fused_gelu_quant"
        if "modulation_quant" in config.enabled:
            from robotics_kernels.ampere_ada.integer import (
                IntegerLinear,
                IntegerProjectionSlice,
            )

            if model.net.training:
                raise ValueError(
                    "Modulation quantization is an inference-only optimization"
                )
            for name, module in model.net.named_modules():
                if not isinstance(module, dit.Block):
                    continue
                boundaries = (
                    (
                        "layer_norm_self_attn",
                        [
                            module.self_attn.q_proj,
                            module.self_attn.k_proj,
                            module.self_attn.v_proj,
                        ],
                    ),
                    ("layer_norm_cross_attn", [module.cross_attn.q_proj]),
                    ("layer_norm_mlp", [module.mlp.layer1]),
                )
                for norm_name, consumers in boundaries:
                    if all(
                        isinstance(m, (IntegerLinear, IntegerProjectionSlice))
                        for m in consumers
                    ):
                        formats = tuple(sorted({m.bits for m in consumers}))
                        patches.set(
                            getattr(module, norm_name), "_robotics_quant_bits", formats
                        )
                        report["coverage"][name + "." + norm_name + ".quant"] = {
                            "formats": list(formats),
                            "backend": "fused_modulation_pack",
                        }
        if "cuda_graph" in config.enabled:
            import torch
            from torch.utils import _pytree
            from robotics_kernels.graph import CudaGraphCall

            original = model.net.forward
            graph = None
            previous_spec = None
            previous_constants = None

            def forward(module, *args, **kwargs):
                nonlocal graph, previous_spec, previous_constants
                leaves, spec = _pytree.tree_flatten((args, kwargs))
                tensor_indices = [
                    i for i, x in enumerate(leaves) if isinstance(x, torch.Tensor)
                ]
                if any(not leaves[i].is_cuda for i in tensor_indices):
                    raise ValueError("DiT graph requires CUDA tensor inputs")
                constants = tuple(
                    (i, x) for i, x in enumerate(leaves) if i not in tensor_indices
                )
                if (
                    graph is None
                    or previous_spec != spec
                    or constants != previous_constants
                ):

                    def execute(*values):
                        combined = leaves[:]
                        for i, x in zip(tensor_indices, values):
                            combined[i] = x
                        a, k = _pytree.tree_unflatten(combined, spec)
                        return original(*a, **k)

                    graph = CudaGraphCall(execute)
                    previous_spec = spec
                    previous_constants = constants
                result = graph(*(leaves[i] for i in tensor_indices))
                report["graph"] = {"captures": graph.captures, "replays": graph.replays}
                return result

            patches.bind(model.net, "forward", forward)
        yield report
    finally:
        stack.close()
        patches.restore()


def _tier_sites(model, selector, tier):
    if tier is None:
        return None
    import torch
    from .progressive import cosmos_precision_map

    candidates = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and selector(n) == "dit"
    ]
    if any(m.weight.dtype != torch.bfloat16 for _, m in candidates):
        raise ValueError(
            "Progressive tiers require all candidate Linear weights to start as BF16"
        )
    names = [n for n, _ in candidates]
    return cosmos_precision_map(names, tier)
