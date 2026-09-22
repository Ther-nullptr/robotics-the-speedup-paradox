"""Bounded BF16 CUTLASS convolution candidate with explicit layout costs.

Weights are packed once to KTRSC. Each complete call converts NCHW/NCDHW
activations to NDHWC and returns NCHW/NCDHW in the original input memory
format; ``forward_packed`` excludes activation-layout conversions. Conv2d
uses a singleton depth.
No causal padding or cache behavior is implemented here: callers retain their
existing wrapper and pass its already-padded tensor to the convolution.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path


TACTICS = {
    0: {"threadblock": (128, 128, 32), "warp": (64, 64, 32), "stages": 3},
    1: {"threadblock": (128, 64, 32), "warp": (64, 32, 32), "stages": 3},
    2: {"threadblock": (64, 64, 64), "warp": (32, 32, 64), "stages": 3},
    3: {"threadblock": (256, 64, 32), "warp": (64, 32, 32), "stages": 3},
    # Explicit generic candidates motivated by the measured C=Kout=96 family.
    # N32 avoids an output-channel tail; larger TileK trades fewer iterator
    # advances for masked channel work. Tactics 4/7 isolate pipeline depth.
    4: {"threadblock": (128, 32, 32), "warp": (64, 32, 32), "stages": 3},
    5: {"threadblock": (128, 32, 64), "warp": (64, 32, 64), "stages": 3},
    6: {"threadblock": (64, 32, 64), "warp": (32, 32, 64), "stages": 3},
    7: {"threadblock": (128, 32, 32), "warp": (64, 32, 32), "stages": 4},
}


def convolution_plan(
    weight_shape, *, stride, padding, dilation, groups=1, padding_mode="zeros", tactic=0
):
    """Validate the supported mathematical operation without importing Torch."""
    dimensions = len(weight_shape) - 2
    if dimensions not in (2, 3):
        raise ValueError("Only Conv2d and Conv3d weights are supported")
    if groups != 1:
        raise ValueError("CUTLASS convolution requires groups=1")
    if padding_mode != "zeros":
        raise ValueError("Only zero padding mode is supported")
    if tactic not in TACTICS:
        raise ValueError("Unknown convolution tactic")
    if any(int(v) <= 0 for v in weight_shape):
        raise ValueError("Convolution weight dimensions must be positive")
    if weight_shape[0] % 8 or weight_shape[1] % 8:
        raise ValueError("Input and output channels must be a multiple of 8")
    if any(k > 32 for k in weight_shape[2:]):
        raise ValueError("Convolution kernel dimensions must be at most 32")

    def values(value, name, minimum):
        if isinstance(value, int):
            value = (value,) * dimensions
        if isinstance(value, str) or len(value) != dimensions:
            raise ValueError(f"Explicit {dimensions}D {name} is required")
        if any(not isinstance(v, int) or v < minimum or v > 32 for v in value):
            raise ValueError(f"Invalid convolution {name}")
        return tuple(value)

    stride = values(stride, "stride", 1)
    padding = values(padding, "padding", 0)
    dilation = values(dilation, "dilation", 1)
    if any(d != 1 for d in dilation):
        raise ValueError("Only unit dilation is supported")
    return {
        "dimensions": dimensions,
        "in_channels": weight_shape[1],
        "out_channels": weight_shape[0],
        "kernel_3d": ((1,) if dimensions == 2 else ()) + tuple(weight_shape[2:]),
        "stride_3d": ((1,) if dimensions == 2 else ()) + stride,
        "padding_3d": ((0,) if dimensions == 2 else ()) + padding,
        "tactic": tactic,
    }


def build_inputs():
    """Resolve only the separately maintained package-owned CUDA and headers."""
    package = Path(__file__).resolve().parent
    dependency = package / "third_party/cutlass"
    sources = [package / "csrc/convolution_fprop.cu"]
    includes = [dependency / "include", dependency / "tools/util/include"]
    provenance = dependency / "PROVENANCE.json"
    paths = [*sources, *includes, provenance]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError(
            "Owned convolution sources or CUTLASS headers are missing"
        )
    if any(not path.resolve().is_relative_to(package) for path in paths):
        raise ValueError("Convolution build inputs must reside inside robotics")
    return {
        "sources": [str(path) for path in sources],
        "include_paths": [str(path) for path in includes],
        "cutlass_revision": json.loads(provenance.read_text())["revision"],
    }


@lru_cache(maxsize=1)
def load_convolution_extension():
    import torch

    def validated_ops():
        if (
            not hasattr(torch.ops.robotics_convolution, "fprop")
            or not hasattr(torch.ops.robotics_convolution, "tactic_count")
            or torch.ops.robotics_convolution.tactic_count() != len(TACTICS)
        ):
            raise RuntimeError("Restart the process after updating convolution tactics")
        return torch.ops.robotics_convolution

    if hasattr(torch.ops.robotics_convolution, "fprop"):
        return validated_ops()
    from torch.utils.cpp_extension import load

    inputs = build_inputs()
    os.environ.setdefault("MAX_JOBS", "2")
    load(
        name="robotics_convolution_ext",
        sources=inputs["sources"],
        extra_include_paths=inputs["include_paths"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        is_python_module=False,
        verbose=False,
    )
    return validated_ops()


@dataclass
class PackedConvolution:
    """Inference-only immutable-weight pack; tactic remains an explicit choice."""

    weight: object
    bias: object
    plan: dict
    backend: str = "cutlass_sm80_bf16_conv3d"

    @classmethod
    def from_module(cls, module, *, tactic=0):
        import torch

        weight = module.weight
        if module.training:
            raise ValueError("CUTLASS convolution is inference-only; call eval() first")
        if not weight.is_cuda or weight.dtype != torch.bfloat16:
            raise ValueError("CUTLASS convolution requires CUDA BF16 weights")
        if torch.cuda.get_device_capability(weight.device)[0] != 8:
            raise ValueError("CUTLASS convolution candidate supports SM80-SM89 only")
        plan = convolution_plan(
            tuple(weight.shape),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            padding_mode=module.padding_mode,
            tactic=tactic,
        )
        weight = weight.detach()
        if plan["dimensions"] == 2:
            weight = weight.unsqueeze(2)
        packed = weight.permute(0, 2, 3, 4, 1).contiguous()
        bias = module.bias
        if bias is not None:
            if bias.device != weight.device or bias.dtype != weight.dtype:
                raise ValueError(
                    "Convolution bias must share the CUDA BF16 weight format"
                )
            bias = bias.detach().contiguous()
        return cls(packed, bias, plan)

    def pack_input(self, x):
        import torch

        if (
            x.ndim != self.plan["dimensions"] + 2
            or x.shape[1] != self.plan["in_channels"]
            or x.dtype != torch.bfloat16
            or x.device != self.weight.device
        ):
            raise ValueError("Convolution input must match the CUDA BF16 module shape")
        if torch.is_grad_enabled() and x.requires_grad:
            raise ValueError("CUTLASS convolution does not implement autograd")
        if x.ndim == 4:
            x = x.unsqueeze(2)
        return x.permute(0, 2, 3, 4, 1).contiguous()

    def forward_packed(self, x):
        """Return NDHWC output; input/output layout conversion is excluded."""
        return load_convolution_extension().fprop(
            x,
            self.weight,
            self.bias,
            list(self.plan["stride_3d"]),
            list(self.plan["padding_3d"]),
            self.plan["tactic"],
        )

    def __call__(self, x):
        import torch

        memory_format = torch.channels_last if x.ndim == 4 else torch.channels_last_3d
        channels_last = x.is_contiguous(memory_format=memory_format)
        result = self.forward_packed(self.pack_input(x)).permute(0, 4, 1, 2, 3)
        if self.plan["dimensions"] == 2:
            result = result.squeeze(2)
        return result if channels_last else result.contiguous()
