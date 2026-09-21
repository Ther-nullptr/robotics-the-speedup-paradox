"""Resolve owned integer sources and their pinned, packaged CUTLASS headers."""

import json
from pathlib import Path


def build_inputs():
    package = Path(__file__).resolve().parent
    dependency = package / "third_party/cutlass"
    sources = [package / "csrc/integer_gemm.cu"]
    includes = [dependency / "include", dependency / "tools/util/include"]
    if any(
        not p.exists() for p in [*sources, *includes, dependency / "PROVENANCE.json"]
    ):
        raise FileNotFoundError(
            "Robotics integer build sources or packaged CUTLASS headers are missing"
        )
    if any(not p.resolve().is_relative_to(package) for p in [*sources, *includes]):
        raise ValueError("Integer build inputs must reside inside the robotics package")
    provenance = json.loads((dependency / "PROVENANCE.json").read_text())
    return {
        "sources": [str(p) for p in sources],
        "include_paths": [str(p) for p in includes],
        "cutlass_revision": provenance["revision"],
    }
