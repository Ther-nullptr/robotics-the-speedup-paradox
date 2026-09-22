"""Optional Wan VAE normalization/affine/SiLU fusion, local to one model."""

from contextlib import contextmanager
from .patching import Patches


@contextmanager
def optimize_vae_pointwise(model, *, fuse_silu=True):
    import torch
    from robotics_kernels.common.vae import channel_norm_affine, prepare_silu

    if model.training:
        raise ValueError("VAE pointwise fusion is inference-only")
    sites = {
        name: module
        for name, module in model.named_modules()
        if type(module).__name__ == "RMS_norm"
        and type(module).__module__.startswith(
            "cosmos_policy._src.predict2.tokenizers.wan2pt"
        )
    }
    if not sites:
        raise ValueError("No supported Wan VAE normalization modules found")
    paired = {}
    if fuse_silu:
        for parent in model.modules():
            if isinstance(parent, torch.nn.Sequential):
                children = list(parent.children())
                for first, second in zip(children, children[1:]):
                    if any(first is module for module in sites.values()) and isinstance(
                        second, torch.nn.SiLU
                    ):
                        paired[id(first)] = second
    patches = Patches()
    report = {
        "backend": "native_channel_reduce_fused_affine",
        "modules": {},
        "calls": 0,
    }
    try:
        for name, module in sites.items():
            if not module.channel_first or module.gamma.dtype != torch.bfloat16:
                raise ValueError(
                    "Wan VAE fusion requires channel-first BF16 normalization"
                )
            activate = id(module) in paired
            if activate:
                prepare_silu(module.gamma.device)

                def identity(silu, value):
                    return value

                patches.bind(paired[id(module)], "forward", identity)

            def forward(norm, x, activate=activate):
                report["calls"] += 1
                return channel_norm_affine(
                    x, norm.gamma, norm.bias, scale=norm.scale, activate=activate
                )

            patches.bind(module, "forward", forward)
            report["modules"][name] = {"silu": activate}
        yield report
    finally:
        patches.restore()
