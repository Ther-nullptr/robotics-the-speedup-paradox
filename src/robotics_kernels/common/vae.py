"""Fuse channel-normalization pointwise work while retaining Torch's reduction."""

import torch
import triton
import triton.language as tl

_SILU_TABLES = {}


def prepare_silu(device):
    device = torch.device(device)
    if device not in _SILU_TABLES:
        values = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16)
        _SILU_TABLES[device] = torch.nn.functional.silu(values.view(torch.bfloat16))
    return _SILU_TABLES[device]


@triton.jit
def _vae_norm_affine(
    X,
    NORM,
    GAMMA,
    BIAS,
    TABLE,
    OUT,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    X3: tl.constexpr,
    X4: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    O3: tl.constexpr,
    O4: tl.constexpr,
    TOTAL: tl.constexpr,
    CHANNEL_LAST: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SCALAR_BIAS: tl.constexpr,
    ACTIVATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < TOTAL
    if CHANNEL_LAST:
        c = i % C
        pixel = i // C
        b = pixel // (T * H * W)
        t = pixel // (H * W) % T
        h = pixel // W % H
        w = pixel % W
    else:
        w = i % W
        h = i // W % H
        t = i // (W * H) % T
        c = i // (W * H * T) % C
        b = i // (W * H * T * C)
        pixel = b * T * H * W + t * H * W + h * W + w
    xi = b * X0 + c * X1 + t * X2 + h * X3 + w * X4
    oi = b * O0 + c * O1 + t * O2 + h * O3 + w * O4
    x = tl.load(X + xi, mask, 0)
    norm = tl.load(NORM + pixel, mask, 1).to(tl.float32)
    value = tl.div_rn(x.to(tl.float32), norm).to(x.dtype).to(tl.float32)
    value = (value * SCALE).to(x.dtype).to(tl.float32)
    gamma = tl.load(GAMMA + c, mask, 1).to(tl.float32)
    value = (value * gamma).to(x.dtype).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS + c, mask, 0).to(tl.float32)
    else:
        bias = SCALAR_BIAS
    value = (value + bias).to(x.dtype)
    if ACTIVATE:
        index = value.to(tl.uint16, bitcast=True).to(tl.int32)
        value = tl.load(TABLE + index)
    tl.store(OUT + oi, value, mask)


def channel_norm_affine(x, gamma, bias=0.0, *, scale=1.0, activate=False):
    if x.ndim not in (4, 5) or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("VAE fusion requires CUDA BF16 NCHW or NCTHW")
    if min(x.shape) < 1 or gamma.shape != (x.shape[1], *([1] * (x.ndim - 2))):
        raise ValueError("Channel affine shape mismatch")
    has_bias = isinstance(bias, torch.Tensor)
    for value in (gamma, bias) if has_bias else (gamma,):
        if (
            value.device != x.device
            or value.dtype != x.dtype
            or not value.is_contiguous()
        ):
            raise ValueError(
                "Channel affine parameters must be contiguous and match input"
            )
    if has_bias and bias.shape != gamma.shape:
        raise ValueError("Channel bias shape mismatch")
    table = prepare_silu(x.device) if activate else gamma
    norm = x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).contiguous()
    out = torch.empty_like(x)
    shape = (x.shape[0], x.shape[1], 1, *x.shape[2:]) if x.ndim == 4 else x.shape

    def strides(value):
        s = value.stride()
        return (s[0], s[1], 0, *s[2:]) if value.ndim == 4 else s

    channel_last = x.is_contiguous(
        memory_format=torch.channels_last if x.ndim == 4 else torch.channels_last_3d
    )
    with torch.cuda.device(x.device):
        _vae_norm_affine[(triton.cdiv(x.numel(), 256),)](
            x,
            norm,
            gamma,
            bias if has_bias else gamma,
            table,
            out,
            *shape[1:],
            *strides(x),
            *strides(out),
            x.numel(),
            channel_last,
            float(scale),
            has_bias,
            0.0 if has_bias else float(bias),
            activate,
            256,
            enable_fp_fusion=False,
        )
    return out
