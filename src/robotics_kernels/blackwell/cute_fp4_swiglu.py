"""CUTLASS CuTe DSL FP4 GEMM + SwiGLU + FP4-output runtime for Thor."""

from __future__ import annotations

import os
from typing import Any

import torch


_COMPILED: dict[tuple[Any, ...], Any] = {}
_SCALARS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_THOR_COMPAT_INSTALLED = False


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _parse_pair(name: str, default: tuple[int, int]) -> tuple[int, int]:
    raw = os.environ.get(name)
    if not raw:
        return default
    normalized = raw.lower().replace(",", "x")
    try:
        first, second = normalized.split("x", 1)
        return int(first), int(second)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must use MxN syntax, got {raw!r}") from exc


def _install_thor_cutlass_compat() -> None:
    """Enable SM110 in the CUTLASS 4.6 Python checks used by this kernel."""

    global _THOR_COMPAT_INSTALLED
    if _THOR_COMPAT_INSTALLED:
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 11:
        _THOR_COMPAT_INSTALLED = True
        return

    from cutlass.base_dsl.arch import Arch
    from cutlass.cute.nvgpu.tcgen05 import CtaGroup
    from cutlass.cute.nvgpu.tcgen05.copy import _S2TCopyBase
    from cutlass.cute.nvgpu.tcgen05.mma import BlockScaledMmaOp

    sm110a = Arch.sm_110a
    if sm110a not in BlockScaledMmaOp.admissible_archs:
        BlockScaledMmaOp.admissible_archs.append(sm110a)

    def _sm110_s2t_post_init(self) -> None:
        if not isinstance(self.cta_group, CtaGroup):
            raise TypeError("cta_group must be a tcgen05.CtaGroup instance")

    _S2TCopyBase.__post_init__ = _sm110_s2t_post_init
    _THOR_COMPAT_INSTALLED = True


def _make_ptr(tensor: torch.Tensor, dtype, alignment: int):
    import cutlass.cute as cute

    return cute.runtime.make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=alignment,
    )


def _scalars(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SCALARS:
        _SCALARS[index] = (
            torch.ones(1, dtype=torch.float32, device=device),
            torch.ones(1, dtype=torch.float32, device=device),
        )
    return _SCALARS[index]


def _tactic() -> tuple[tuple[int, int], tuple[int, int], bool]:
    tile = _parse_pair("ROBOTICS_CUTE_SWIGLU_TILE", (128, 256))
    cluster = _parse_pair("ROBOTICS_CUTE_SWIGLU_CLUSTER", (2, 1))
    prefetch = os.environ.get("ROBOTICS_CUTE_SWIGLU_PREFETCH", "0") == "1"
    return tile, cluster, prefetch


def _compile(
    packed_input: torch.Tensor,
    packed_weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    packed_output: torch.Tensor,
    output_scale: torch.Tensor,
    alpha: torch.Tensor,
    norm_const: torch.Tensor,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from .cute_fp4_swiglu_kernel import (
        Sm100BlockScaledPersistentDenseGemmSwigluFusionKernel,
    )

    _install_thor_cutlass_compat()
    tile, cluster, prefetch = _tactic()
    device_index = packed_input.device.index
    key = (device_index, tile, cluster, prefetch)
    if key in _COMPILED:
        return _COMPILED[key]

    m = packed_input.shape[0]
    n = packed_weight.shape[0]
    k = packed_input.shape[1] * 2
    sf_m = _round_up(m, 128)
    sf_n = _round_up(n, 128)
    sf_k = _round_up(k // 16, 4)
    n_out = n // 2

    a_ptr = _make_ptr(packed_input, cutlass.Float4E2M1FN, 32)
    b_ptr = _make_ptr(packed_weight, cutlass.Float4E2M1FN, 32)
    a_sf_ptr = _make_ptr(input_scale, cutlass.Float8E4M3FN, 16)
    b_sf_ptr = _make_ptr(weight_scale, cutlass.Float8E4M3FN, 16)
    c_ptr = _make_ptr(packed_output, cutlass.Float4E2M1FN, 32)
    sfc_ptr = _make_ptr(output_scale, cutlass.Float8E4M3FN, 16)
    alpha_tensor = cute.runtime.from_dlpack(alpha)
    norm_tensor = cute.runtime.from_dlpack(norm_const)
    stream = cuda.CUstream(torch.cuda.current_stream(packed_input.device).cuda_stream)

    kernel = Sm100BlockScaledPersistentDenseGemmSwigluFusionKernel(
        sf_vec_size=16,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
        vectorized_f32=True,
        use_prefetch=prefetch,
    )
    if not kernel.can_implement(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.Float4E2M1FN,
        tile,
        cluster,
        m,
        n,
        k,
        1,
        "k",
        "k",
        "n",
    ):
        raise RuntimeError(
            f"CUTLASS SwiGLU tactic tile={tile}, cluster={cluster} cannot "
            f"implement M={m}, N={n}, K={k}"
        )
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
        cluster[0] * cluster[1]
    )
    compiled = cute.compile(
        kernel.wrapper_fp4out,
        m,
        n,
        k,
        sf_m // 128,
        sf_n // 128,
        sf_k // 4,
        sf_m // 128,
        _round_up(n_out // 16, 4) // 4,
        1,
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        c_ptr,
        sfc_ptr,
        alpha_tensor,
        norm_tensor,
        max_active_clusters,
        stream,
        options="--opt-level 2",
    )
    _COMPILED[key] = compiled
    return compiled


def cutlass_swiglu_fp4_packed(
    packed_input: torch.Tensor,
    input_scale: torch.Tensor,
    packed_interleaved_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run W4A4 gate/up GEMM and return packed W4A4 SwiGLU activations."""

    if (
        packed_input.dtype != torch.uint8
        or packed_interleaved_weight.dtype != torch.uint8
    ):
        raise TypeError("CUTLASS FP4 operands must be packed uint8 tensors")
    if input_scale.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
        raise TypeError("CUTLASS FP4 scale factors must be uint8 tensors")
    if packed_input.ndim != 2 or packed_interleaved_weight.ndim != 2:
        raise ValueError("packed operands must be rank-2")
    tensors = (
        packed_input,
        input_scale,
        packed_interleaved_weight,
        weight_scale,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("CUTLASS FP4 operands and scale factors must be CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("CUTLASS FP4 operands and scale factors must share a device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("CUTLASS FP4 operands and scale factors must be contiguous")

    m = packed_input.shape[0]
    if m < 128:
        raise ValueError("CUTLASS fused FP4-output SwiGLU requires at least 128 rows")
    n = packed_interleaved_weight.shape[0]
    k = packed_input.shape[1] * 2
    if packed_interleaved_weight.shape[1] != packed_input.shape[1]:
        raise ValueError("packed gate/up weight K dimension does not match the input")
    if n % 128 or k % 32:
        raise ValueError(f"fused SwiGLU requires N%128=0 and K%32=0, got N={n}, K={k}")

    tile, cluster, _ = _tactic()
    n_out = n // 2
    padded_m = _round_up(m, tile[0] * cluster[0])
    packed_output = torch.empty(
        (padded_m, n_out // 2), dtype=torch.uint8, device=packed_input.device
    )
    sf_m = _round_up(m, 128)
    sf_n_cols = _round_up(n_out // 16, 4)
    sf_m_sfc = _round_up(padded_m, 128)
    output_scale = torch.empty(
        sf_m_sfc * sf_n_cols, dtype=torch.uint8, device=packed_input.device
    )
    expected_a_sf = sf_m * _round_up(k // 16, 4)
    expected_b_sf = _round_up(n, 128) * _round_up(k // 16, 4)
    if input_scale.numel() < expected_a_sf:
        raise ValueError("input scale-factor buffer is too small")
    if weight_scale.numel() < expected_b_sf:
        raise ValueError("weight scale-factor buffer is too small")

    alpha, norm_const = _scalars(packed_input.device)
    compiled = _compile(
        packed_input,
        packed_interleaved_weight,
        input_scale.reshape(-1),
        weight_scale.reshape(-1),
        packed_output,
        output_scale,
        alpha,
        norm_const,
    )

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    stream = cuda.CUstream(torch.cuda.current_stream(packed_input.device).cuda_stream)
    compiled(
        m,
        n,
        k,
        sf_m // 128,
        _round_up(n, 128) // 128,
        _round_up(k // 16, 4) // 4,
        sf_m // 128,
        sf_n_cols // 4,
        _make_ptr(packed_input, cutlass.Float4E2M1FN, 32),
        _make_ptr(packed_interleaved_weight, cutlass.Float4E2M1FN, 32),
        _make_ptr(input_scale, cutlass.Float8E4M3FN, 16),
        _make_ptr(weight_scale, cutlass.Float8E4M3FN, 16),
        _make_ptr(packed_output, cutlass.Float4E2M1FN, 32),
        _make_ptr(output_scale, cutlass.Float8E4M3FN, 16),
        cute.runtime.from_dlpack(alpha),
        cute.runtime.from_dlpack(norm_const),
        stream,
    )
    expected_output_sf = sf_m * sf_n_cols
    return packed_output[:m], output_scale[:expected_output_sf]


def prepare_cutlass_swiglu_fp4(
    packed_interleaved_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    in_features: int,
) -> None:
    """Compile the dynamic CuTe kernel before the measured inference region."""

    tile, cluster, prefetch = _tactic()
    device_index = packed_interleaved_weight.device.index
    if (device_index, tile, cluster, prefetch) in _COMPILED:
        return
    rows = max(128, tile[0] * cluster[0])
    packed_input = torch.zeros(
        (rows, in_features // 2),
        dtype=torch.uint8,
        device=packed_interleaved_weight.device,
    )
    input_scale = torch.full(
        (_round_up(rows, 128), _round_up(in_features // 16, 4)),
        0x38,
        dtype=torch.uint8,
        device=packed_interleaved_weight.device,
    )
    cutlass_swiglu_fp4_packed(
        packed_input,
        input_scale,
        packed_interleaved_weight,
        weight_scale,
    )
    torch.cuda.synchronize(packed_interleaved_weight.device)
