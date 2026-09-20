"""Locally owned CUTLASS MXFP8 pack and GEMM backend."""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path

from robotics_kernels.blackwell.extension_loader import (
    load_extension,
    operator_namespace,
)


OP_NAMESPACE = "robotics_mxfp8"
EXTENSION_ENV = "ROBOTICS_MXFP8_SO"
DEFAULT_EXTENSION_PATH = (
    Path(__file__).resolve().parents[3]
    / "build"
    / "robotics_mxfp8"
    / "robotics_mxfp8_ext.so"
)
_LOAD_ERROR: str | None = None


_namespace = partial(operator_namespace, OP_NAMESPACE)


def load_mxfp8_extension(
    path: str | os.PathLike[str] | None = None,
) -> bool:
    """Load this backend while retaining its independent policy and error state."""
    global _LOAD_ERROR
    loaded, _LOAD_ERROR = load_extension(
        OP_NAMESPACE,
        ("pack", "linear"),
        lambda: [
            Path(path).expanduser()
            if path
            else Path(
                os.environ.get(EXTENSION_ENV, DEFAULT_EXTENSION_PATH)
            ).expanduser()
        ],
        missing_description="missing MXFP8 operators",
    )
    return loaded


def mxfp8_ops():
    if not load_mxfp8_extension():
        raise RuntimeError(f"MXFP8 extension is unavailable: {_LOAD_ERROR}")
    return _namespace()


def scale_storage_size(rows: int, columns: int) -> int:
    return ((int(rows) + 127) // 128) * ((int(columns) + 127) // 128) * 512
