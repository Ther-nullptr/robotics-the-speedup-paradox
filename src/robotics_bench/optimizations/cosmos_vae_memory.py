"""Owned adaptation of Wan VAE implicit spatial padding, inspired by worldmodel.

Only the symmetric spatial zeros move into the convolution descriptor. Temporal
cache concatenation and left padding retain the Cosmos causal-wrapper contract.
No worldmodel runtime import or cross-request cache is introduced.
"""

from contextlib import contextmanager
from .patching import Patches


@contextmanager
def optimize_vae_spatial_padding(encoder):
    import torch
    import torch.nn.functional as functional

    if encoder.training:
        raise ValueError("VAE spatial padding is an inference-only optimization")
    sites = [
        (name, module)
        for name, module in encoder.named_modules()
        if isinstance(module, torch.nn.Conv3d)
        and type(module).__name__ == "CausalConv3d"
        and type(module).__module__.startswith("cosmos_policy.")
    ]
    if not sites:
        raise ValueError("No supported Cosmos causal convolution modules found")
    for name, module in sites:
        padding = module._padding
        if (
            len(padding) != 6
            or padding[0] != padding[1]
            or padding[2] != padding[3]
            or padding[5] != 0
            or min(padding) < 0
            or tuple(module.padding) != (0, 0, 0)
            or module.padding_mode != "zeros"
        ):
            raise ValueError(
                f"{name}: expected symmetric spatial padding and causal left padding"
            )
    patches = Patches()
    report = {"scope": "encoder", "modules": {}, "cross_request_cache": False}
    try:
        for name, module in sites:
            padding = module._padding
            row = {
                "calls": 0,
                "temporal_pad_calls": 0,
                "skipped_explicit_pad_calls": 0,
                "implicit_spatial_padding": [padding[2], padding[0]],
            }
            report["modules"][name] = row
            patches.set(module, "padding", (0, padding[2], padding[0]))

            def forward(conv, x, cache_x=None, row=row):
                row["calls"] += 1
                temporal = conv._padding[4]
                if cache_x is not None and temporal > 0:
                    x = torch.cat([cache_x.to(x.device), x], dim=2)
                    temporal -= cache_x.shape[2]
                if temporal:
                    x = functional.pad(x, (0, 0, 0, 0, temporal, 0))
                    row["temporal_pad_calls"] += 1
                else:
                    row["skipped_explicit_pad_calls"] += 1
                return conv._conv_forward(x, conv.weight, conv.bias)

            patches.bind(module, "forward", forward)
        yield report
    finally:
        patches.restore()
