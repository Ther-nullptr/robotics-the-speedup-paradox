"""Pointwise fusion preserving native low-precision rounding boundaries.

Design follows the VLM fused operators, independently implemented for robotics.
Triton is loaded only when this optional module is imported. Reduction order and
GEMM precision are unchanged. GELU uses an exact finite-input 16-bit lookup table
built with the active Torch implementation before graph capture.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope(
    X,
    C,
    S,
    OUT,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < TOTAL
    d = i % D
    t = i // D % L
    h = i // (D * L) % H
    b = i // (D * L * H)
    off = b * X0 + h * X1 + t * X2
    paired = tl.where(d < D // 2, d + D // 2, d - D // 2)
    x = tl.load(X + off + d, mask, 0)
    r = tl.load(X + off + paired, mask, 0).to(tl.float32)
    r = tl.where(d < D // 2, -r, r)
    c = tl.load(C + (b * L + t) * D + d, mask, 0).to(tl.float32)
    s = tl.load(S + (b * L + t) * D + d, mask, 0).to(tl.float32)
    a = (x.to(tl.float32) * c).to(x.dtype).to(tl.float32)
    v = (r * s).to(x.dtype).to(tl.float32)
    tl.store(OUT + i, (a + v).to(x.dtype), mask)


@triton.jit
def _residual(
    X, Y, G, OUT, T: tl.constexpr, D: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    x = tl.load(X + i, mask, 0)
    y = tl.load(Y + i, mask, 0).to(tl.float32)
    gate = tl.load(G + i // (T * D) * D + i % D, mask, 0).to(tl.float32)
    product = (y * gate).to(x.dtype).to(tl.float32)
    tl.store(OUT + i, (x.to(tl.float32) + product).to(x.dtype), mask)


@triton.jit
def _gelu_mul(X, U, LUT, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    x = tl.load(X + i, mask, 0)
    index = x.to(tl.uint16, bitcast=True).to(tl.int32)
    activated = tl.load(LUT + index).to(tl.float32)
    up = tl.load(U + i, mask, 0).to(tl.float32)
    tl.store(OUT + i, (activated * up).to(x.dtype), mask)


_LUTS = {}


def prepare_gelu(device, dtype, approximate="tanh"):
    key = (torch.device(device), dtype, approximate)
    if key not in _LUTS:
        values = (
            torch.arange(65536, device=device, dtype=torch.int32)
            .to(torch.int16)
            .view(dtype)
        )
        _LUTS[key] = torch.nn.functional.gelu(values, approximate=approximate)
    return _LUTS[key]


def _lowp(*tensors):
    first = tensors[0]
    if first.dtype not in (torch.float16, torch.bfloat16) or not first.is_cuda:
        raise ValueError("Expected CUDA float16 or bfloat16 tensors")
    if any(t.device != first.device or t.dtype != first.dtype for t in tensors):
        raise ValueError("All tensors must share device and dtype")


def rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    _lowp(q, k, cos, sin)
    if unsqueeze_dim != 1 or q.ndim != 4 or k.ndim != 4 or q.shape[-1] % 2:
        raise ValueError("RoPE requires B,H,L,D and an even head dimension")
    if (
        q.shape[0] != k.shape[0]
        or q.shape[2:] != k.shape[2:]
        or cos.shape != (q.shape[0], q.shape[2], q.shape[3])
        or sin.shape != cos.shape
    ):
        raise ValueError("Incompatible RoPE shapes")
    if (
        q.stride(-1) != 1
        or k.stride(-1) != 1
        or not cos.is_contiguous()
        or not sin.is_contiguous()
    ):
        raise ValueError("Unsupported RoPE layout")
    results = []
    with torch.cuda.device(q.device):
        for x in (q, k):
            out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            _rope[(triton.cdiv(x.numel(), 256),)](
                x,
                cos,
                sin,
                out,
                x.shape[1],
                x.shape[2],
                x.shape[3],
                *x.stride()[:3],
                x.numel(),
                256,
                enable_fp_fusion=False,
            )
            results.append(out)
    return tuple(results)


def gated_residual(x, y, gate):
    if x is None or y is None:
        return y if x is None else x
    if gate is None:
        return x + y
    _lowp(x, y, gate)
    if (
        x.ndim != 3
        or y.shape != x.shape
        or gate.shape != (x.shape[0], 1, x.shape[2])
        or not all(t.is_contiguous() for t in (x, y, gate))
    ):
        raise ValueError("Gated residual expects contiguous B,T,D and B,1,D")
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _residual[(triton.cdiv(x.numel(), 256),)](
            x,
            y,
            gate,
            out,
            x.shape[1],
            x.shape[2],
            x.numel(),
            256,
            enable_fp_fusion=False,
        )
    return out


def gelu_mul(gate, up):
    _lowp(gate, up)
    if gate.shape != up.shape or not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("GELU inputs must have matching contiguous layouts")
    lut = prepare_gelu(gate.device, gate.dtype)
    out = torch.empty_like(gate)
    with torch.cuda.device(gate.device):
        _gelu_mul[(triton.cdiv(gate.numel(), 256),)](
            gate, up, lut, out, gate.numel(), 256, enable_fp_fusion=False
        )
    return out


@triton.jit
def _norm_affine(
    X,
    INV,
    S,
    H,
    OUT,
    D: tl.constexpr,
    T: tl.constexpr,
    STRIDE: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    x = tl.load(X + i, valid, 0).to(tl.float32)
    inv = tl.load(INV + i // D, valid, 0).to(tl.float32)
    off = i % D
    if ADAPTIVE:
        off = off + i // (T * D) * STRIDE
    scale = tl.load(S + off, valid, 0).to(tl.float32)
    value = x * inv
    value = value * (1.0 + scale)
    if ADAPTIVE:
        shift = tl.load(H + off, valid, 0).to(tl.float32)
        value = value + shift
    tl.store(OUT + i, value, valid)


def norm_affine(x, inv, scale, shift=None):
    _lowp(x)
    if x.ndim != 3 or not x.is_contiguous() or inv.shape != (*x.shape[:2], 1):
        raise ValueError("Norm affine requires contiguous B,T,D and B,T,1 inverse RMS")
    expected = (x.shape[0], 1, x.shape[2]) if shift is not None else (x.shape[2],)
    if scale.shape != expected or (shift is not None and shift.shape != expected):
        raise ValueError("Incompatible norm scale/shift shape")
    for tensor in (inv, scale, shift):
        if tensor is not None and (
            tensor.device != x.device
            or tensor.dtype != torch.float32
            or tensor.stride(-1) != 1
        ):
            raise ValueError(
                "Norm statistics and modulation must be float32 on the input device"
            )
    if shift is not None and scale.stride() != shift.stride():
        raise ValueError("Modulation layouts must match")
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _norm_affine[(triton.cdiv(x.numel(), 256),)](
            x,
            inv,
            scale,
            shift if shift is not None else scale,
            out,
            x.shape[2],
            x.shape[1],
            scale.stride(0),
            shift is not None,
            x.numel(),
            256,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _modulate(
    X, S, H, OUT, T: tl.constexpr, D: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    x = tl.load(X + i, valid, 0)
    off = i // (T * D) * D + i % D
    scale = tl.load(S + off, valid, 0).to(tl.float32)
    shift = tl.load(H + off, valid, 0).to(tl.float32)
    scale = (1.0 + scale).to(x.dtype).to(tl.float32)
    product = (x.to(tl.float32) * scale).to(x.dtype).to(tl.float32)
    tl.store(OUT + i, (product + shift).to(x.dtype), valid)


def modulate(x, scale, shift):
    _lowp(x, scale, shift)
    if (
        x.ndim != 3
        or scale.shape != (x.shape[0], 1, x.shape[2])
        or shift.shape != scale.shape
        or not all(t.is_contiguous() for t in (x, scale, shift))
    ):
        raise ValueError("Modulation expects contiguous B,T,D and B,1,D")
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _modulate[(triton.cdiv(x.numel(), 256),)](
            x,
            scale,
            shift,
            out,
            x.shape[1],
            x.shape[2],
            x.numel(),
            256,
            enable_fp_fusion=False,
        )
    return out
