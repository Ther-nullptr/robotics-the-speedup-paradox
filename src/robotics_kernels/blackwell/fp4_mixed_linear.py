"""Loader for the CUTLASS FP4-weight W4A16 experimental extension."""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path

import torch

from robotics_kernels.blackwell.extension_loader import (
    load_extension,
    operator_namespace,
)


OP_NAMESPACE = "robotics_cutlass_fp4_mixed"
EXTENSION_ENV = "ROBOTICS_CUTLASS_FP4_MIXED_SO"
DEFAULT_EXTENSION_PATH = (
    Path(__file__).resolve().parents[3]
    / "build"
    / "robotics_cutlass_fp4_mixed"
    / "robotics_cutlass_fp4_mixed_ext.so"
)
REQUIRED_OPS = (
    "supports_device",
    "repack_weight_scales",
    "materialize_weight_bf16",
    "materialize_weight_bf16_out",
    "linear_w4a16",
    "linear_w4a16_tactic",
)
_LOAD_ERROR: str | None = None


_ops_namespace = partial(operator_namespace, OP_NAMESPACE)


def load_cutlass_fp4_mixed_extension(
    path: str | os.PathLike[str] | None = None,
) -> bool:
    """Load this backend while retaining its independent policy and error state."""
    global _LOAD_ERROR
    loaded, _LOAD_ERROR = load_extension(
        OP_NAMESPACE,
        REQUIRED_OPS,
        lambda: [
            Path(path).expanduser()
            if path
            else Path(
                os.environ.get(EXTENSION_ENV, DEFAULT_EXTENSION_PATH)
            ).expanduser()
        ],
    )
    return loaded


def require_cutlass_fp4_mixed_extension(
    device: torch.device | int | None = None,
) -> None:
    if not load_cutlass_fp4_mixed_extension():
        raise RuntimeError(f"CUTLASS FP4 mixed extension is unavailable: {_LOAD_ERROR}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
    major, minor = torch.cuda.get_device_capability(index)
    if not bool(_ops_namespace().supports_device(major * 10 + minor)):
        raise RuntimeError(f"device capability {major}.{minor} is unsupported")


def mixed_ops():
    require_cutlass_fp4_mixed_extension()
    return _ops_namespace()
