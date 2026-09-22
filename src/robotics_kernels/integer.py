"""Compatibility imports; implementation lives in the Ampere/Ada backend."""

from .ampere_ada.integer import (
    IntegerLinear,
    PackedActivation,
    load_integer_extension,
    prepare_activation,
)

__all__ = [
    "IntegerLinear",
    "PackedActivation",
    "load_integer_extension",
    "prepare_activation",
]
