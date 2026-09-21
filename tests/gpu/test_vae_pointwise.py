"""VAE fusion must retain native BF16 rounding, including norm and SiLU."""

import os
import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)
import torch


@pytest.mark.parametrize("shape", [(2, 24, 9, 13), (1, 96, 3, 11, 9)])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "strided"])
@pytest.mark.parametrize("activate", [False, True])
def test_native_channel_reduction_and_rounding(shape, layout, activate):
    from robotics_kernels.common.vae import channel_norm_affine

    torch.manual_seed(17)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    if layout == "channels_last":
        fmt = torch.channels_last if x.ndim == 4 else torch.channels_last_3d
        x = x.contiguous(memory_format=fmt)
    elif layout == "strided":
        x = x[..., ::2]
    gamma = torch.randn(
        (shape[1], *([1] * (x.ndim - 2))), device=x.device, dtype=x.dtype
    )
    bias = torch.randn_like(gamma)
    scale = shape[1] ** 0.5
    expected = torch.nn.functional.normalize(x, dim=1) * scale * gamma + bias
    if activate:
        expected = torch.nn.functional.silu(expected)
    actual = channel_norm_affine(x, gamma, bias, scale=scale, activate=activate)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("magnitude", [0.0, 1e-20, 1e-13])
def test_normalize_epsilon_and_scalar_bias(magnitude):
    from robotics_kernels.common.vae import channel_norm_affine

    x = torch.full((1, 8, 3, 7), magnitude, device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones((8, 1, 1), device=x.device, dtype=x.dtype)
    expected = torch.nn.functional.normalize(x, dim=1) * (8**0.5) * gamma + 0.0
    assert torch.equal(channel_norm_affine(x, gamma, scale=8**0.5), expected)
