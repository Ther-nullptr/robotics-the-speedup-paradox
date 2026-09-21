"""Independent Cosmos DiT switches with native normalization preserved."""

from contextlib import contextmanager, ExitStack
from .patching import Patches


def _modulation(x, norm, scale, shift, enabled):
    y = norm(x)
    if not enabled:
        return y * (1 + scale) + shift
    from robotics_kernels.fused import modulate

    b, t, h, w, d = x.shape
    return modulate(
        y.reshape(b * t, h * w, d),
        scale.reshape(b * t, 1, d).contiguous(),
        shift.reshape(b * t, 1, d).contiguous(),
    ).reshape_as(x)


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
        "modulation",
        "gated_residual",
        "cuda_graph",
        "shared_quant",
        "activation_quant_fusion",
        "integer_grouped",
        "integer_pack_reuse",
        "integer_qkv",
        "integer_group_views",
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

    names = [
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and selector(n) == "dit"
    ]
    return cosmos_precision_map(names, tier)
