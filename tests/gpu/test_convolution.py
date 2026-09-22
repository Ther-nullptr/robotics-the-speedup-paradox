"""CUTLASS convolution and causal wrapper semantics on real CUDA tensors."""

import os
import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)
import torch


@pytest.mark.parametrize("tactic", range(8))
@pytest.mark.parametrize("dimensions,bias", [(2, True), (3, True), (3, False)])
def test_cutlass_convolution_matches_native_on_current_stream(tactic, dimensions, bias):
    from robotics_kernels.ampere_ada.convolution import PackedConvolution

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        cls = torch.nn.Conv2d if dimensions == 2 else torch.nn.Conv3d
        module = cls(
            16,
            24,
            3,
            stride=2,
            padding=1,
            bias=bias,
            device="cuda",
            dtype=torch.bfloat16,
        ).eval()
        shape = (2, 16, 15, 17) if dimensions == 2 else (1, 16, 5, 15, 17)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        packed = PackedConvolution.from_module(module, tactic=tactic)
        actual = packed(x)
        expected = module(x)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert actual.is_contiguous()
    assert actual.shape == expected.shape


def test_causal_cache_padding_and_patch_restoration():
    from robotics_bench.optimizations.cosmos_convolution import optimize_convolutions

    class CausalConv3d(torch.nn.Conv3d):
        def forward(self, x, cache_x=None):
            if cache_x is not None:
                x = torch.cat((cache_x, x), 2)
            left = 2 if cache_x is None else 2 - cache_x.shape[2]
            return super().forward(torch.nn.functional.pad(x, (1, 1, 1, 1, left, 0)))

    model = torch.nn.ModuleDict(
        {
            "causal": CausalConv3d(16, 24, 3, device="cuda", dtype=torch.bfloat16),
            "rgb": torch.nn.Conv3d(3, 16, 3, device="cuda", dtype=torch.bfloat16),
            "grouped": torch.nn.Conv2d(
                16, 16, 3, groups=2, device="cuda", dtype=torch.bfloat16
            ),
        }
    ).eval()
    x = torch.randn(1, 16, 2, 8, 9, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn_like(x[:, :, :1])
    original = model["causal"]._conv_forward
    with torch.inference_mode():
        expected = [model["causal"](x), model["causal"](x, cache)]
        with optimize_convolutions(model, tactic=1) as report:
            actual = [model["causal"](x), model["causal"](x, cache)]
            assert report["modules"]["causal"]["backend"] == "cutlass_sm80_bf16_conv3d"
            assert report["modules"]["causal"]["calls"] == 2
            assert report["modules"]["rgb"]["backend"] == "native"
            assert report["modules"]["grouped"]["backend"] == "native"
        assert model["causal"]._conv_forward == original
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.02)


def test_convolution_rejects_cpu_before_build_and_invalid_packed_layout():
    from robotics_kernels.ampere_ada.convolution import PackedConvolution

    module = torch.nn.Conv3d(8, 8, 1, dtype=torch.bfloat16).eval()
    with pytest.raises(ValueError, match="CUDA BF16"):
        PackedConvolution.from_module(module)
    with torch.inference_mode():
        module = module.cuda()
        packed = PackedConvolution.from_module(module)
        x = torch.randn(1, 3, 4, 5, 8, device="cuda", dtype=torch.bfloat16)
        with pytest.raises((ValueError, RuntimeError), match="contiguous"):
            packed.forward_packed(x.transpose(1, 2))


@pytest.mark.parametrize("tactic", [4, 5, 6, 7])
def test_narrow_output_tactics_cover_c96_channel_tail_and_bias(tactic):
    from robotics_kernels.ampere_ada.convolution import (
        PackedConvolution,
        load_convolution_extension,
    )

    with torch.inference_mode():
        module = torch.nn.Conv3d(96, 96, 3, device="cuda", dtype=torch.bfloat16).eval()
        x = torch.randn(1, 96, 4, 30, 32, device="cuda", dtype=torch.bfloat16)
        x = x.contiguous(memory_format=torch.channels_last_3d)
        actual = PackedConvolution.from_module(module, tactic=tactic)(x)
        torch.testing.assert_close(actual, module(x), rtol=0.02, atol=0.02)
        assert load_convolution_extension().tactic_count() == 8


@pytest.mark.parametrize("dimensions", [2, 3])
def test_convolution_preserves_channels_last_without_output_copy(dimensions):
    from robotics_kernels.ampere_ada.convolution import PackedConvolution

    cls = torch.nn.Conv2d if dimensions == 2 else torch.nn.Conv3d
    memory_format = torch.channels_last if dimensions == 2 else torch.channels_last_3d
    shape = (1, 16, 8, 9) if dimensions == 2 else (1, 16, 3, 8, 9)
    with torch.inference_mode():
        module = cls(16, 24, 3, padding=1, device="cuda", dtype=torch.bfloat16).eval()
        packed = PackedConvolution.from_module(module)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous(
            memory_format=memory_format
        )
        actual = packed(x)
        assert actual.is_contiguous(memory_format=memory_format)
        torch.testing.assert_close(actual, module(x), rtol=0.02, atol=0.02)


def test_c96_policy_keeps_contiguous_layout_native_and_records_fallback():
    from robotics_bench.optimizations.cosmos_convolution import optimize_convolutions

    module = torch.nn.Conv3d(96, 96, 3, device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randn(1, 96, 3, 12, 14, device="cuda", dtype=torch.bfloat16)
    channels_last = x.contiguous(memory_format=torch.channels_last_3d)
    with torch.inference_mode():
        expected = module(x)
        with optimize_convolutions(module, policy="c96") as report:
            native = module(x)
            candidate = module(channels_last)
            row = report["modules"][""]
            assert row["calls"] == 1
            assert row["native_fallback_calls"] == 1
            assert row["native_fallback_reasons"] == {
                "c96 policy requires channels_last_3d input": 1
            }
        assert torch.equal(native, expected)
        torch.testing.assert_close(candidate, expected, rtol=0.02, atol=0.02)
