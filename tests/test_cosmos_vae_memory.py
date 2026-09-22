"""Spatial padding migration must preserve temporal history and restore modules."""

import pytest

torch = pytest.importorskip("torch")


class CausalConv3d(torch.nn.Conv3d):
    def __init__(self):
        super().__init__(4, 8, 3, padding=0)
        self._padding = (1, 1, 1, 1, 2, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and padding[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(torch.nn.functional.pad(x, padding))


# A small fixture implements the same known Cosmos causal-wrapper contract.
CausalConv3d.__module__ = "cosmos_policy._src.predict2.tokenizers.wan2pt1"


@pytest.mark.parametrize("previous_frames", [0, 1, 2])
def test_spatial_padding_preserves_causal_cache_and_restores(previous_frames):
    from robotics_bench.optimizations.cosmos_vae_memory import (
        optimize_vae_spatial_padding,
    )

    torch.manual_seed(9)
    module = CausalConv3d().eval()
    x = torch.randn(1, 4, 2, 7, 9)
    cache = torch.randn(1, 4, previous_frames, 7, 9) if previous_frames else None
    expected = module(x, cache)
    original = module.forward
    with optimize_vae_spatial_padding(module) as report:
        assert module.padding == (0, 1, 1)
        actual = module(x, cache)
        row = report["modules"][""]
        assert row["calls"] == 1
        assert row["skipped_explicit_pad_calls"] == int(previous_frames == 2)
    torch.testing.assert_close(actual, expected)
    assert module.padding == (0, 0, 0)
    assert module.forward == original


def test_asymmetric_spatial_padding_is_rejected_without_partial_changes():
    from robotics_bench.optimizations.cosmos_vae_memory import (
        optimize_vae_spatial_padding,
    )

    module = CausalConv3d().eval()
    module._padding = (0, 1, 1, 1, 2, 0)
    with pytest.raises(ValueError, match="symmetric"):
        with optimize_vae_spatial_padding(module):
            pass
    assert module.padding == (0, 0, 0)
