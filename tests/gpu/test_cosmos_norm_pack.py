"""Bounded numerical and graph checks for fused non-affine AdaLN packing."""

import os
import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)
import torch


def unpack(pack):
    if pack.bits == 8:
        return pack.data.int()
    low = pack.data.int() & 15
    high = pack.data.int() >> 4
    values = torch.stack((low, high), -1).flatten(1)
    return torch.where(values >= 8, values - 16, values)


@pytest.mark.parametrize("bits,k", [(4, 258), (8, 258), (4, 2048), (8, 2048)])
def test_norm_pack_reference_and_graph_replay(bits, k):
    from robotics_kernels.ampere_ada import modulation as module

    assert hasattr(module, "prepare_norm_modulation"), "Fused LayerNorm packing missing"
    torch.manual_seed(731)
    x = torch.randn(2, 17, k, device="cuda", dtype=torch.bfloat16)
    x[0, 0].zero_()
    x[1, 0].fill_(32)
    scale = torch.randn(2, 1, k, device="cuda", dtype=x.dtype)
    shift = torch.randn_like(scale)
    norm = torch.nn.LayerNorm(k, eps=1e-6, elementwise_affine=False).cuda().eval()

    def run():
        return module.prepare_norm_modulation(x, scale, shift, bits, eps=norm.eps)

    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for offset in (0.0, 0.25):
        x.add_(offset)
        graph.replay()
        expected = module.prepare_modulation(norm(x), scale, shift, bits)
        torch.testing.assert_close(
            actual.scales, expected.scales, rtol=0.008, atol=1e-7
        )
        delta = (unpack(actual) - unpack(expected)).abs()
        assert delta.max().item() <= 1
        assert (delta != 0).float().mean().item() <= 0.002
        assert not unpack(actual)[:, k:].count_nonzero().item()
        assert actual.shape == expected.shape


def test_norm_pack_model_boundary_skips_native_norm():
    from robotics_bench.optimizations.cosmos import _modulation
    from robotics_kernels.ampere_ada import modulation as module

    assert hasattr(module, "prepare_norm_modulation"), "Fused LayerNorm packing missing"
    x = torch.randn(1, 2, 3, 4, 128, device="cuda", dtype=torch.bfloat16)
    norm = torch.nn.LayerNorm(128, eps=1e-6, elementwise_affine=False).cuda().eval()
    norm._robotics_quant_bits = (4, 8)
    norm._robotics_norm_quant = True

    def reject_native(*_):
        raise AssertionError("Native LayerNorm must not execute in the fused boundary")

    norm.register_forward_pre_hook(reject_native)
    scale = torch.zeros(1, 2, 1, 1, 128, device="cuda", dtype=x.dtype)
    packed = _modulation(x, norm, scale, scale, True)
    assert set(packed.packs) == {4, 8}
    assert packed.for_bits(4).shape == tuple(x.shape[:-1])


@pytest.mark.parametrize("bits", [4, 8])
def test_residual_norm_pack_preserves_residual_and_normalized_pack(bits):
    from robotics_kernels.ampere_ada import modulation as module

    assert hasattr(module, "prepare_residual_norm_modulation"), (
        "Residual fusion missing"
    )
    torch.manual_seed(89)
    x = torch.randn(2, 17, 2048, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x)
    gate = torch.randn(2, 1, 2048, device="cuda", dtype=x.dtype)
    scale, shift = torch.randn_like(gate), torch.randn_like(gate)
    expected_residual = x + gate * y
    expected_pack = module.prepare_norm_modulation(
        expected_residual, scale, shift, bits, eps=1e-6
    )
    residual, packed = module.prepare_residual_norm_modulation(
        x, y, gate, scale, shift, bits, eps=1e-6
    )
    assert torch.equal(residual, expected_residual)
    assert torch.equal(packed.data, expected_pack.data)
    assert torch.equal(packed.scales, expected_pack.scales)
