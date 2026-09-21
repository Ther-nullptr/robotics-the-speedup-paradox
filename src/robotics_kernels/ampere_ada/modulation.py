"""BF16 AdaLN modulation directly into symmetric integer activation buffers."""

from dataclasses import dataclass, replace
import math
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .integer import PackedActivation


@triton.jit
def _pack_modulated_row(
    x,
    row,
    SCALE,
    SHIFT,
    Q,
    S,
    K: tl.constexpr,
    PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.arange(0, BLOCK)
    valid = index < K
    offset = row // TOKENS * K + index
    scale = tl.load(SCALE + offset, valid, 0).to(tl.float32)
    shift = tl.load(SHIFT + offset, valid, 0).to(tl.float32)
    factor = (1.0 + scale).to(x.dtype).to(tl.float32)
    product = (x.to(tl.float32) * factor).to(x.dtype).to(tl.float32)
    value = (product + shift).to(x.dtype).to(tl.float32)
    limit: tl.constexpr = (1 << (BITS - 1)) - 1
    maximum = tl.max(tl.abs(value), 0)
    quant_scale = tl.where(maximum > 0, maximum * (1.0 / limit), 1.0)
    tl.store(S + row, quant_scale)
    q = libdevice.nearbyint(tl.div_rn(value, quant_scale))
    q = tl.minimum(tl.maximum(q, -limit), limit).to(tl.int32)
    if BITS == 8:
        tl.store(Q + row * PAD + index, q.to(tl.int8), index < PAD)
    else:
        pairs = tl.reshape(q & 15, (BLOCK // 2, 2))
        packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1).to(tl.uint8)
        pair_index = tl.arange(0, BLOCK // 2)
        tl.store(Q + row * (PAD // 2) + pair_index, packed, pair_index < PAD // 2)


@triton.jit
def _prepare_modulation(
    X,
    SCALE,
    SHIFT,
    Q,
    S,
    K: tl.constexpr,
    PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + index, index < K, 0)
    _pack_modulated_row(x, row, SCALE, SHIFT, Q, S, K, PAD, TOKENS, BITS, BLOCK)


@triton.jit
def _normalize_pack_row(
    x,
    row,
    SCALE,
    SHIFT,
    Q,
    S,
    K: tl.constexpr,
    PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    index = tl.arange(0, BLOCK)
    valid = index < K
    value = x.to(tl.float32)
    mean = tl.sum(value, 0) / K
    centered = tl.where(valid, value - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / K
    # Match the native materialized BF16 boundary before modulation. Reduction
    # order still differs from native LayerNorm; this is not bitwise guaranteed.
    normalized = (centered * tl.rsqrt(variance + EPS)).to(x.dtype)
    _pack_modulated_row(
        normalized, row, SCALE, SHIFT, Q, S, K, PAD, TOKENS, BITS, BLOCK
    )


@triton.jit
def _prepare_norm_modulation(
    X,
    SCALE,
    SHIFT,
    Q,
    S,
    K: tl.constexpr,
    PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + index, index < K, 0)
    _normalize_pack_row(x, row, SCALE, SHIFT, Q, S, K, PAD, TOKENS, BITS, BLOCK, EPS)


@triton.jit
def _prepare_residual_norm_modulation(
    X,
    SCALE,
    SHIFT,
    Q,
    S,
    K: tl.constexpr,
    PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    Y,
    GATE,
    RESIDUAL,
):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    valid = index < K
    x = tl.load(X + row * K + index, valid, 0)
    y = tl.load(Y + row * K + index, valid, 0).to(tl.float32)
    gate = tl.load(GATE + row // TOKENS * K + index, valid, 0).to(tl.float32)
    product = (y * gate).to(x.dtype).to(tl.float32)
    residual = (x.to(tl.float32) + product).to(x.dtype)
    tl.store(RESIDUAL + row * K + index, residual, valid)
    _normalize_pack_row(
        residual, row, SCALE, SHIFT, Q, S, K, PAD, TOKENS, BITS, BLOCK, EPS
    )


def prepare_modulation(x, scale, shift, bits):
    return _prepare(x, scale, shift, bits)


def prepare_norm_modulation(x, scale, shift, bits, *, eps):
    """Fuse non-affine LayerNorm into packing; FP32 reduction order may differ."""
    if eps is None:
        raise ValueError("LayerNorm epsilon must be finite and positive")
    return _prepare(x, scale, shift, bits, eps=eps)


def prepare_residual_norm_modulation(x, y, gate, scale, shift, bits, *, eps):
    """Return the exact BF16 residual and its fused normalized integer pack."""
    if eps is None:
        raise ValueError("LayerNorm epsilon must be finite and positive")
    return _prepare(x, scale, shift, bits, eps=eps, residual=(y, gate))


def _prepare(x, scale, shift, bits, *, eps=None, residual=None):
    if eps is not None:
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("LayerNorm epsilon must be finite and positive")
        eps = float(eps)
    if (
        bits not in (4, 8)
        or x.ndim != 3
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        raise ValueError(
            "Modulation packing requires contiguous CUDA BF16 B,T,D and bits=4 or 8"
        )
    b, tokens, k = x.shape
    if min(b, tokens, k) < 1 or k > 65536:
        raise ValueError("Unsupported modulation dimensions")
    for value in (scale, shift):
        if (
            value.shape != (b, 1, k)
            or value.dtype != x.dtype
            or value.device != x.device
        ):
            raise ValueError(
                "Modulation scale and shift must match B,1,D, dtype and device"
            )
    scale, shift = scale.contiguous(), shift.contiguous()
    if residual is not None:
        y, gate = residual
        if (
            y.shape != x.shape
            or y.dtype != x.dtype
            or y.device != x.device
            or not y.is_contiguous()
            or gate.shape != (b, 1, k)
            or gate.dtype != x.dtype
            or gate.device != x.device
        ):
            raise ValueError("Residual operands must match BF16 input and B,1,D gate")
        gate = gate.contiguous()
        residual_out = torch.empty_like(x)
    rows, padded = b * tokens, triton.cdiv(k, 128) * 128
    data = torch.empty(
        (rows, padded if bits == 8 else padded // 2),
        device=x.device,
        dtype=torch.int8 if bits == 8 else torch.uint8,
    )
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        kernel = _prepare_modulation if eps is None else _prepare_norm_modulation
        extra = {} if eps is None else {"EPS": eps}
        if residual is not None:
            kernel = _prepare_residual_norm_modulation
            extra.update(Y=y, GATE=gate, RESIDUAL=residual_out)
        kernel[(rows,)](
            x,
            scale,
            shift,
            data,
            scales,
            k,
            padded,
            tokens,
            bits,
            triton.next_power_of_2(padded),
            num_warps=4 if padded <= 2048 else 8,
            enable_fp_fusion=False,
            **extra,
        )
    packed = PackedActivation(data, scales, (b, tokens), k, bits)
    return packed if residual is None else (residual_out, packed)


@dataclass
class PackedActivations:
    """Internal model boundary carrying only the requested integer formats."""

    packs: dict[int, PackedActivation]
    _robotics_packed_input = True

    def for_bits(self, bits):
        if bits not in self.packs:
            raise ValueError("A consumer requested an unprepared integer format")
        return self.packs[bits]

    def with_leading_shape(self, shape):
        shape = tuple(shape)
        if (
            not shape
            or min(shape) < 1
            or any(
                math.prod(shape) != pack.data.shape[0] for pack in self.packs.values()
            )
        ):
            raise ValueError("Packed leading shape must preserve the row count")
        return PackedActivations(
            {bits: replace(pack, shape=shape) for bits, pack in self.packs.items()}
        )
