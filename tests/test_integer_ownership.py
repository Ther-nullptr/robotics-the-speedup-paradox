"""Integer builds must resolve implementation and headers inside this package."""

from pathlib import Path
import importlib
import json
import os
import sys


def test_integer_build_inputs_are_owned_and_ignore_external_project_paths(monkeypatch):
    package = Path(__file__).parents[1] / "src/robotics_kernels/ampere_ada"
    assert (package / "build.py").is_file(), "Owned integer build inputs are missing"
    before = set(sys.modules)
    monkeypatch.setenv("ROBOTICS_CUTLASS_ROOT", "/an/external/project/must/not/be/read")
    api = importlib.import_module("robotics_kernels.ampere_ada.build")
    inputs = api.build_inputs()
    assert all(
        Path(p).resolve().is_relative_to(package.resolve())
        for p in inputs["sources"] + inputs["include_paths"]
    )
    assert inputs["cutlass_revision"] == "982748aa7356fa838c2ea4994ddcb0b2a4b4cefa"
    assert "torch" not in set(sys.modules) - before
    assert (package / "third_party/cutlass/LICENSE.txt").is_file()
    assert (
        len(
            json.loads((package / "third_party/cutlass/PROVENANCE.json").read_text())[
                "files"
            ]
        )
        > 800
    )
    assert (
        os.environ["ROBOTICS_CUTLASS_ROOT"] == "/an/external/project/must/not/be/read"
    )
