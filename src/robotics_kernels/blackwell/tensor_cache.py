"""Tensor identity shared by activation and fused-projection caches."""

from __future__ import annotations

from typing import Any


def tensor_cache_key(tensor: Any) -> tuple[Any, ...]:
    try:
        version = tensor._version
    except RuntimeError:  # Inference tensors do not expose a version counter.
        version = None
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        version,
    )
