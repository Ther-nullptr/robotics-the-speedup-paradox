"""Local fused SwiGLU producer for CUTLASS-compatible MXFP8 activations."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from robotics_kernels.blackwell.extension_loader import load_extension


OP_NAMESPACE = "robotics_mxfp8_swiglu"
EXTENSION_ENV = "ROBOTICS_MXFP8_SWIGLU_SO"
DEFAULT_EXTENSION_PATH = (
    Path(__file__).resolve().parents[3]
    / "build"
    / "robotics_mxfp8_swiglu"
    / "robotics_mxfp8_swiglu_ext.so"
)
_LOAD_ERROR: str | None = None


def load_mxfp8_swiglu_extension(
    path: str | os.PathLike[str] | None = None,
) -> bool:
    """Load this backend while retaining its independent policy and error state."""
    global _LOAD_ERROR
    loaded, _LOAD_ERROR = load_extension(
        OP_NAMESPACE,
        ("silu_mul_pack",),
        lambda: [
            Path(path).expanduser()
            if path
            else Path(
                os.environ.get(EXTENSION_ENV, DEFAULT_EXTENSION_PATH)
            ).expanduser()
        ],
        missing_description="missing silu_mul_pack operator",
    )
    return loaded


def make_bf16_silu_table(device: torch.device | str = "cuda") -> torch.Tensor:
    raw = torch.arange(65536, device=device, dtype=torch.int32).to(torch.uint16)
    values = raw.view(torch.bfloat16)
    return torch.nn.functional.silu(values).contiguous()


def silu_mul_pack_mxfp8(
    gate_up: torch.Tensor,
    intermediate_size: int,
    silu_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not load_mxfp8_swiglu_extension():
        raise RuntimeError(f"MXFP8 SwiGLU extension is unavailable: {_LOAD_ERROR}")
    output = torch.ops.robotics_mxfp8_swiglu.silu_mul_pack(
        gate_up, int(intermediate_size), silu_table
    )
    return output[0], output[1]
