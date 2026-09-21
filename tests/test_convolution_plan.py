"""Supported convolution boundaries and package-owned build inputs."""

from importlib import import_module
from pathlib import Path
import sys

import pytest


def api():
    path = Path(__file__).parents[1] / "src/robotics_kernels/ampere_ada/convolution.py"
    assert path.is_file(), "Owned CUTLASS convolution candidate is missing"
    return import_module("robotics_kernels.ampere_ada.convolution")


def test_convolution_plan_promotes_2d_without_temporal_padding():
    plan = api().convolution_plan(
        (192, 96, 3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1)
    )
    assert plan["dimensions"] == 2
    assert plan["kernel_3d"] == (1, 3, 3)
    assert plan["stride_3d"] == (1, 2, 2)
    assert plan["padding_3d"] == (0, 0, 0)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"groups": 2}, "groups"),
        ({"dilation": (2, 1, 1)}, "dilation"),
        ({"padding_mode": "reflect"}, "padding mode"),
        ({"stride": (0, 1, 1)}, "stride"),
        ({"padding": (-1, 0, 0)}, "padding"),
        ({"tactic": 99}, "tactic"),
    ],
)
def test_convolution_plan_rejects_unsupported_semantics(overrides, match):
    args = {"stride": (1, 1, 1), "padding": (0, 0, 0), "dilation": (1, 1, 1)}
    args.update(overrides)
    with pytest.raises(ValueError, match=match):
        api().convolution_plan((96, 96, 3, 3, 3), **args)


def test_convolution_plan_does_not_silently_pad_three_channel_inputs():
    with pytest.raises(ValueError, match="multiple of 8"):
        api().convolution_plan(
            (96, 3, 3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)
        )


def test_convolution_build_uses_owned_headers_without_importing_torch(monkeypatch):
    before = set(sys.modules)
    monkeypatch.setenv("ROBOTICS_CUTLASS_ROOT", "/not/a/valid/external/tree")
    inputs = api().build_inputs()
    package = Path(__file__).parents[1] / "src/robotics_kernels/ampere_ada"
    assert inputs["sources"] == [str(package / "csrc/convolution_fprop.cu")]
    assert all(Path(p).is_relative_to(package) for p in inputs["include_paths"])
    assert inputs["cutlass_revision"] == "982748aa7356fa838c2ea4994ddcb0b2a4b4cefa"
    assert "torch" not in set(sys.modules) - before


def test_tactic_description_is_explicit_and_not_model_shape_dispatch():
    tactics = api().TACTICS
    assert set(tactics) == set(range(8))
    assert tactics[0]["threadblock"] == (128, 128, 32)
    assert tactics[4]["threadblock"] == (128, 32, 32)
    assert tactics[5]["threadblock"] == (128, 32, 64)
    assert tactics[6]["threadblock"] == (64, 32, 64)
    assert tactics[7]["threadblock"] == tactics[4]["threadblock"]
    assert tactics[7]["stages"] == 4
    # The generic validation retains support beyond the tuning family C=96.
    for tactic in tactics:
        plan = api().convolution_plan(
            (40, 24, 3, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            dilation=(1, 1, 1),
            tactic=tactic,
        )
        assert plan["tactic"] == tactic


def test_in_process_extension_from_earlier_tactic_set_requires_restart(monkeypatch):
    from types import SimpleNamespace

    module = api()
    old_ops = SimpleNamespace(fprop=object(), tactic_count=lambda: 4)
    fake_torch = SimpleNamespace(ops=SimpleNamespace(robotics_convolution=old_ops))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "torch.utils.cpp_extension", SimpleNamespace(load=None)
    )
    module.load_convolution_extension.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="Restart"):
            module.load_convolution_extension()
    finally:
        module.load_convolution_extension.cache_clear()


def test_loader_rejects_stale_binary_returned_after_a_failed_jit_build(monkeypatch):
    from types import SimpleNamespace

    module = api()
    ops = SimpleNamespace()
    fake_torch = SimpleNamespace(ops=SimpleNamespace(robotics_convolution=ops))

    def load(**kwargs):
        # A prior failed JIT build can leave an older binary in the cache.
        ops.fprop = object()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "torch.utils.cpp_extension", SimpleNamespace(load=load)
    )
    module.load_convolution_extension.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="Restart"):
            module.load_convolution_extension()
    finally:
        module.load_convolution_extension.cache_clear()


def test_c96_selection_is_explicit_and_rejects_unmeasured_module_families():
    module = import_module("robotics_bench.optimizations.cosmos_convolution")
    assert hasattr(module, "selection_reason"), "Explicit C96 module policy is missing"
    assert module.selection_reason((96, 96, 3, 3, 3), (1, 1, 1), "c96") is None
    assert module.selection_reason((192, 192, 3, 3, 3), (1, 1, 1), "c96")
    assert module.selection_reason((96, 96, 3, 3), (1, 1), "c96")
    assert module.selection_reason((96, 96, 3, 3, 3), (2, 1, 1), "c96")
    with pytest.raises(ValueError, match="policy"):
        module.selection_reason((96, 96, 3, 3, 3), (1, 1, 1), "automatic")
