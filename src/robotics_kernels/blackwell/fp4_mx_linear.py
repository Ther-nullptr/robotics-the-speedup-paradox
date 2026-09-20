"""Loader for the CUTLASS MXFP8 x MXFP4 W4A8 extension."""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path

import torch

from robotics_kernels.blackwell.extension_loader import (
    load_extension,
    operator_namespace,
)


OP_NAMESPACE = "robotics_cutlass_fp4_mx"
EXTENSION_ENV = "ROBOTICS_CUTLASS_FP4_MX_SO"
DEFAULT_EXTENSION_PATH = (
    Path(__file__).resolve().parents[3]
    / "build/robotics_cutlass_fp4_mx"
    / "robotics_cutlass_fp4_mx_ext.so"
)
REQUIRED_OPS = (
    "supports_device",
    "pack_mxfp4_weight",
    "pack_mxfp8_activation",
    "pack_swiglu_mxfp8",
    "linear_w4a8_packed",
    "linear_w4a8",
)
_LOAD_ERROR: str | None = None


_ops_namespace = partial(operator_namespace, OP_NAMESPACE)


def load_cutlass_fp4_mx_extension(
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


def mx_ops():
    if not load_cutlass_fp4_mx_extension():
        raise RuntimeError(f"CUTLASS FP4 MX extension is unavailable: {_LOAD_ERROR}")
    major, minor = torch.cuda.get_device_capability()
    if not bool(_ops_namespace().supports_device(major * 10 + minor)):
        raise RuntimeError(f"device capability {major}.{minor} is unsupported")
    return _ops_namespace()
