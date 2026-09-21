"""CUTLASS CuTe MXFP8 Gate-Up GEMM with SwiGLU and MXFP8 output."""

from __future__ import annotations

import os
from typing import Any

import torch

from robotics_kernels.blackwell.cute_fp4_swiglu import (
    _install_thor_cutlass_compat,
    _make_ptr,
    _parse_pair,
)


_COMPILED: dict[tuple[Any, ...], Any] = {}
_SCALARS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _round_up(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _scale_size(rows: int, columns: int) -> int:
    return ((rows + 127) // 128) * ((columns + 127) // 128) * 512


def _scalars(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SCALARS:
        _SCALARS[index] = (
            torch.ones(1, dtype=torch.float32, device=device),
            torch.ones(1, dtype=torch.float32, device=device),
        )
    return _SCALARS[index]


def _tactic() -> tuple[tuple[int, int], tuple[int, int], bool]:
    tile = _parse_pair("ROBOTICS_CUTE_MXFP8_SWIGLU_TILE", (256, 256))
    cluster = _parse_pair("ROBOTICS_CUTE_MXFP8_SWIGLU_CLUSTER", (2, 1))
    prefetch = os.environ.get("ROBOTICS_CUTE_MXFP8_SWIGLU_PREFETCH", "0") == "1"
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

    from robotics_kernels.blackwell.cute_fp4_swiglu_kernel import (
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
    k = packed_input.shape[1]
    n_out = n // 2
    a_ptr = _make_ptr(packed_input, cutlass.Float8E4M3FN, 16)
    b_ptr = _make_ptr(packed_weight, cutlass.Float8E4M3FN, 16)
    a_sf_ptr = _make_ptr(input_scale, cutlass.Float8E8M0FNU, 16)
    b_sf_ptr = _make_ptr(weight_scale, cutlass.Float8E8M0FNU, 16)
    c_ptr = _make_ptr(packed_output, cutlass.Float8E4M3FN, 16)
    sfc_ptr = _make_ptr(output_scale, cutlass.Float8E8M0FNU, 16)
    stream = cuda.CUstream(torch.cuda.current_stream(packed_input.device).cuda_stream)
    kernel = Sm100BlockScaledPersistentDenseGemmSwigluFusionKernel(
        sf_vec_size=32,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
        vectorized_f32=True,
        use_prefetch=prefetch,
    )
    if not kernel.can_implement(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.Float8E4M3FN,
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
            f"MXFP8 SwiGLU tactic tile={tile}, cluster={cluster} cannot "
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
        _round_up(m, 128) // 128,
        _round_up(n, 128) // 128,
        _round_up(k // 32, 4) // 4,
        _round_up(packed_output.shape[0], 128) // 128,
        _round_up(n_out // 32, 4) // 4,
        1,
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        c_ptr,
        sfc_ptr,
        cute.runtime.from_dlpack(alpha),
        cute.runtime.from_dlpack(norm_const),
        max_active_clusters,
        stream,
        options="--opt-level 2",
    )
    _COMPILED[key] = compiled
    return compiled


def cutlass_swiglu_mxfp8_packed(
    packed_input: torch.Tensor,
    input_scale: torch.Tensor,
    packed_interleaved_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MXFP8 Gate-Up and return fresh block32 MXFP8 SwiGLU output."""

    if packed_input.dtype != torch.float8_e4m3fn:
        raise TypeError("MXFP8 input must use torch.float8_e4m3fn")
    if packed_interleaved_weight.dtype != torch.float8_e4m3fn:
        raise TypeError("MXFP8 weight must use torch.float8_e4m3fn")
    if input_scale.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
        raise TypeError("MXFP8 scale factors must be uint8 E8M0 payloads")
    tensors = (
        packed_input,
        input_scale,
        packed_interleaved_weight,
        weight_scale,
    )
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("MXFP8 operands must be contiguous CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("MXFP8 operands must share a CUDA device")
    if packed_input.ndim != 2 or packed_interleaved_weight.ndim != 2:
        raise ValueError("MXFP8 operands must be rank two")
    m, k = packed_input.shape
    n = packed_interleaved_weight.shape[0]
    if packed_interleaved_weight.shape[1] != k or n % 256 or k % 32:
        raise ValueError(f"invalid MXFP8 Gate-Up shape M={m}, N={n}, K={k}")
    if m < 128:
        raise ValueError("fused MXFP8 SwiGLU requires at least 128 rows")
    if input_scale.numel() != _scale_size(m, k):
        raise ValueError("input scale buffer has the wrong size")
    if weight_scale.numel() != _scale_size(n, k):
        raise ValueError("weight scale buffer has the wrong size")

    tile, cluster, _ = _tactic()
    padded_m = _round_up(m, tile[0] * cluster[0])
    n_out = n // 2
    packed_output = torch.empty(
        (padded_m, n_out),
        dtype=torch.float8_e4m3fn,
        device=packed_input.device,
    )
    output_scale = torch.empty(
        _scale_size(padded_m, n_out),
        dtype=torch.uint8,
        device=packed_input.device,
    )
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
        _round_up(m, 128) // 128,
        _round_up(n, 128) // 128,
        _round_up(k // 32, 4) // 4,
        _round_up(padded_m, 128) // 128,
        _round_up(n_out // 32, 4) // 4,
        _make_ptr(packed_input, cutlass.Float8E4M3FN, 16),
        _make_ptr(packed_interleaved_weight, cutlass.Float8E4M3FN, 16),
        _make_ptr(input_scale, cutlass.Float8E8M0FNU, 16),
        _make_ptr(weight_scale, cutlass.Float8E8M0FNU, 16),
        _make_ptr(packed_output, cutlass.Float8E4M3FN, 16),
        _make_ptr(output_scale, cutlass.Float8E8M0FNU, 16),
        cute.runtime.from_dlpack(alpha),
        cute.runtime.from_dlpack(norm_const),
        stream,
    )
    return packed_output[:m], output_scale[: _scale_size(m, n_out)]
