"""Compatibility imports for portable fusion operations."""

from .common.fused import (
    rope,
    gated_residual,
    gelu_mul,
    modulate,
    norm_affine,
    prepare_gelu,
)

__all__ = [
    "rope",
    "gated_residual",
    "gelu_mul",
    "modulate",
    "norm_affine",
    "prepare_gelu",
]
