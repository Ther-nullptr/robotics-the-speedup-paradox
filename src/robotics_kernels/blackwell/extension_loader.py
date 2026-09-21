"""Common torch.ops loading mechanics; backend policy stays in each adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import torch


def operator_namespace(name: str):
    return getattr(torch.ops, name, None)


def has_operator(namespace: str, name: str) -> bool:
    return hasattr(operator_namespace(namespace), name)


def load_extension(
    namespace: str,
    required_ops: tuple[str, ...],
    candidate_paths: Callable[[], Iterable[Path]],
    *,
    configure: Callable[[], object] | None = None,
    check_exists: bool = False,
    missing_description: str | None = None,
) -> tuple[bool, str | None]:
    """Load an extension, preserving caller-owned retries and policy setup."""
    if all(has_operator(namespace, name) for name in required_ops):
        try:
            if configure is not None:
                configure()
        except (ValueError, RuntimeError) as error:
            return False, str(error)
        return True, None

    try:
        candidates = candidate_paths()
    except (OSError, RuntimeError) as error:
        return False, str(error)

    last_error: str | None = None
    for candidate in candidates:
        try:
            if check_exists and not candidate.exists():
                last_error = f"{candidate} does not exist"
                continue
            torch.ops.load_library(str(candidate))
        except (OSError, RuntimeError) as error:
            last_error = f"{candidate}: {error}"
            continue
        missing = [name for name in required_ops if not has_operator(namespace, name)]
        if missing:
            detail = missing_description or f"missing operators {missing}"
            last_error = f"{candidate}: {detail}"
            continue
        try:
            if configure is not None:
                configure()
        except (ValueError, RuntimeError) as error:
            last_error = str(error)
            continue
        return True, None
    return False, last_error
