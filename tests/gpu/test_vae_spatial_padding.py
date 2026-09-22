"""Known Cosmos causal wrapper with implicit spatial padding on CUDA."""

import os
import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)
import torch


@pytest.mark.parametrize("previous_frames", [0, 1, 2])
def test_real_causal_convolution_preserves_cache_and_restores(previous_frames):
    from cosmos_policy._src.predict2.tokenizers.wan2pt1 import CausalConv3d
    from robotics_bench.optimizations.cosmos_vae_memory import (
        optimize_vae_spatial_padding,
    )

    module = (
        CausalConv3d(96, 96, 3, padding=1)
        .eval()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    x = torch.randn(1, 96, 2, 9, 11, device="cuda", dtype=torch.bfloat16)
    cache = (
        torch.randn(1, 96, previous_frames, 9, 11, device="cuda", dtype=x.dtype)
        if previous_frames
        else None
    )
    with torch.inference_mode():
        expected = module(x, cache)
        with optimize_vae_spatial_padding(module):
            actual = module(x, cache)
        assert torch.equal(module(x, cache), expected)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


def test_cutlass_plans_see_the_migrated_spatial_padding():
    from cosmos_policy._src.predict2.tokenizers.wan2pt1 import CausalConv3d
    from robotics_bench.optimizations.cosmos_vae_memory import (
        optimize_vae_spatial_padding,
    )
    from robotics_bench.optimizations.cosmos_convolution import optimize_convolutions

    module = (
        CausalConv3d(96, 96, 3, padding=1)
        .eval()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    x = torch.randn(1, 96, 2, 9, 11, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    with torch.inference_mode():
        expected = module(x)
        with (
            optimize_vae_spatial_padding(module),
            optimize_convolutions(module, policy="c96") as report,
        ):
            actual = module(x)
            assert report["modules"][""]["plan"]["padding_3d"] == (0, 1, 1)
            assert report["modules"][""]["calls"] == 1
    assert module.padding == (0, 0, 0)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
