"""Real INT4/INT8 Linear paths with independent packing and fused epilogues.

The lifecycle follows the VLM FP wrappers: weight pack at construction, optional
shared activation preparation, explicit packed-input execution and backend ID.
Formats are symmetric signed row-wise activations and output-channel weights,
float32 scales, int32 accumulation, BF16 output; INT4 uses low-nibble-first bytes.
"""

from dataclasses import dataclass
from functools import lru_cache
import os
import weakref

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@lru_cache(maxsize=1)
def load_integer_extension():
    if hasattr(torch.ops.robotics_integer, "gemm"):
        if not hasattr(torch.ops.robotics_integer, "gemm_biasless"):
            raise RuntimeError(
                "Restart the process after updating the robotics integer extension"
            )
        return torch.ops.robotics_integer
    from .build import build_inputs

    inputs = build_inputs()
    from torch.utils.cpp_extension import load

    os.environ.setdefault("MAX_JOBS", "2")
    load(
        name="robotics_integer_ext",
        sources=inputs["sources"],
        extra_include_paths=inputs["include_paths"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--fmad=false",
            "-lineinfo",
        ],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.robotics_integer


@triton.jit
def _activation_values(
    X,
    UP,
    LUT,
    row,
    idx,
    K: tl.constexpr,
    MODE: tl.constexpr,
    SX: tl.constexpr,
    SU: tl.constexpr,
):
    x = tl.load(X + row * SX + idx, idx < K, 0)
    if MODE != 0:
        index = x.to(tl.uint16, bitcast=True).to(tl.int32)
        value = tl.load(LUT + index).to(tl.float32)
        if MODE == 1:
            up = tl.load(UP + row * SU + idx, idx < K, 0).to(tl.float32)
            value = (value * up).to(x.dtype).to(tl.float32)
        return value
    return x.to(tl.float32)


@triton.jit
def _prepare(
    X,
    UP,
    LUT,
    Q,
    S,
    K: tl.constexpr,
    PAD: tl.constexpr,
    BITS: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    REUSE: tl.constexpr,
    SX: tl.constexpr,
    SU: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    x = _activation_values(X, UP, LUT, row, idx, K, MODE, SX, SU)
    limit: tl.constexpr = (1 << (BITS - 1)) - 1
    maximum = tl.max(tl.abs(x), 0)
    scale = tl.where(maximum > 0, maximum * (1.0 / limit), 1.0)
    tl.store(S + row, scale)
    if BITS == 8:
        q = libdevice.nearbyint(tl.div_rn(x, scale))
        q = tl.minimum(tl.maximum(q, -limit), limit).to(tl.int8)
        tl.store(Q + row * PAD + idx, q, idx < PAD)
    elif REUSE:
        q = libdevice.nearbyint(tl.div_rn(x, scale))
        q = tl.minimum(tl.maximum(q, -limit), limit).to(tl.int32) & 15
        pairs = tl.reshape(q, (BLOCK // 2, 2))
        packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1).to(tl.uint8)
        pair_idx = tl.arange(0, BLOCK // 2)
        tl.store(Q + row * (PAD // 2) + pair_idx, packed, pair_idx < PAD // 2)
    else:
        lo = _activation_values(X, UP, LUT, row, idx * 2, K, MODE, SX, SU)
        hi = _activation_values(X, UP, LUT, row, idx * 2 + 1, K, MODE, SX, SU)
        qlo = tl.minimum(
            tl.maximum(libdevice.nearbyint(tl.div_rn(lo, scale)), -limit), limit
        ).to(tl.int32)
        qhi = tl.minimum(
            tl.maximum(libdevice.nearbyint(tl.div_rn(hi, scale)), -limit), limit
        ).to(tl.int32)
        packed = ((qlo & 15) | ((qhi & 15) << 4)).to(tl.uint8)
        tl.store(Q + row * (PAD // 2) + idx, packed, idx < PAD // 2)


@dataclass
class PackedActivation:
    data: torch.Tensor
    scales: torch.Tensor
    shape: tuple
    logical_k: int
    bits: int
    format_version: str = "symmetric-row-v1"


def prepare_activation(x, bits, *, up=None, gelu=None, pack_reuse=False):
    if (
        bits not in (4, 8)
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.stride(-1) != 1
        or x.ndim < 2
    ):
        raise ValueError(
            "Expected CUDA BF16 rows with unit feature stride and bits=4 or 8"
        )
    mode = 0
    lut = x
    if gelu is not None:
        if gelu not in ("tanh", "none"):
            raise ValueError("Unsupported GELU approximation")
        from ..common.fused import prepare_gelu

        lut = prepare_gelu(x.device, x.dtype, gelu)
        mode = 1 if up is not None else 2
    if up is not None and (
        gelu is None
        or up.shape != x.shape
        or up.device != x.device
        or up.dtype != x.dtype
        or up.stride(-1) != 1
    ):
        raise ValueError("Invalid gated activation input")
    k = x.shape[-1]
    if k < 1 or k > 65536:
        raise ValueError("Unsupported activation dimensions")
    rows = x.numel() // k
    if rows < 1:
        raise ValueError("Unsupported activation dimensions")
    try:
        matrix = x.view(rows, k)
        up_matrix = up.view(rows, k) if up is not None else matrix
    except RuntimeError as error:
        raise ValueError(
            "Activation leading dimensions must form regularly strided rows"
        ) from error
    pad = triton.cdiv(k, 128) * 128
    data = torch.empty(
        (rows, pad if bits == 8 else pad // 2),
        device=x.device,
        dtype=torch.int8 if bits == 8 else torch.uint8,
    )
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        _prepare[(rows,)](
            x,
            up if up is not None else x,
            lut,
            data,
            scales,
            k,
            pad,
            bits,
            mode,
            triton.next_power_of_2(pad),
            pack_reuse,
            matrix.stride(0),
            up_matrix.stride(0),
            num_warps=4 if pad <= 2048 else 8,
            enable_fp_fusion=False,
        )
    return PackedActivation(data, scales, tuple(x.shape[:-1]), k, bits)


class IntegerLinear(torch.nn.Module):
    def __init__(
        self,
        packed,
        scales,
        bias,
        in_features,
        out_features,
        bits,
        tactic=0,
        pack_reuse=False,
        biasless_epilogue=False,
    ):
        super().__init__()
        self.register_buffer("packed_weight", packed)
        self.register_buffer("weight_scale", scales)
        self.register_buffer("bias", bias)
        self.register_buffer(
            "_weight_descriptor",
            torch.empty(0, device=packed.device, dtype=torch.bfloat16),
            persistent=False,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.tactic = tactic
        self.pack_reuse = pack_reuse
        self.biasless_epilogue = biasless_epilogue
        self.backend = f"cutlass_sm80_int{bits}_s32_bf16"
        self._ops = load_integer_extension()

    @property
    def weight(self):
        return self._weight_descriptor

    @classmethod
    def from_linear(
        cls, linear, *, bits=8, tactic=0, pack_reuse=False, biasless_epilogue=False
    ):
        if (
            bits not in (4, 8)
            or not linear.weight.is_cuda
            or linear.weight.dtype != torch.bfloat16
        ):
            raise ValueError("IntegerLinear requires CUDA BF16 weights and bits=4 or 8")
        load_integer_extension()
        w = linear.weight.detach().float()
        if not torch.isfinite(w).all():
            raise ValueError("Weights must be finite")
        limit = 2 ** (bits - 1) - 1
        maximum = w.abs().amax(dim=1)
        scales = torch.where(maximum > 0, maximum / limit, torch.ones_like(maximum))
        values = torch.round(w / scales[:, None]).clamp(-limit, limit).to(torch.int8)
        n, k = w.shape
        npad = triton.cdiv(n, 8) * 8
        kpad = triton.cdiv(k, 128) * 128
        values = torch.nn.functional.pad(values, (0, kpad - k, 0, npad - n))
        scales = torch.nn.functional.pad(scales, (0, npad - n), value=1.0)
        if bits == 4:
            values = (
                (values[:, 0::2].to(torch.int32) & 15)
                | ((values[:, 1::2].to(torch.int32) & 15) << 4)
            ).to(torch.uint8)
        bias = (
            None
            if linear.bias is None
            else torch.nn.functional.pad(
                linear.bias.detach().to(torch.bfloat16), (0, npad - n)
            )
        )
        return cls(
            values.contiguous(),
            scales.contiguous(),
            bias,
            k,
            n,
            bits,
            tactic,
            pack_reuse,
            biasless_epilogue,
        )

    def pack_input(self, x):
        if getattr(x, "_robotics_packed_input", False):
            packed = x.for_bits(self.bits)
            if packed.logical_k != self.in_features:
                raise ValueError("Wrong prepared input feature count")
            return packed
        if x.shape[-1] != self.in_features:
            raise ValueError("Wrong input feature count")
        return prepare_activation(x, self.bits, pack_reuse=self.pack_reuse)

    def forward_packed(self, packed):
        if (
            packed.logical_k != self.in_features
            or packed.bits != self.bits
            or packed.format_version != "symmetric-row-v1"
        ):
            raise ValueError("Incompatible activation pack")
        operation = (
            self._ops.gemm_biasless if self.biasless_epilogue else self._ops.gemm
        )
        output = operation(
            packed.data,
            packed.scales,
            self.packed_weight,
            self.weight_scale,
            self.bias,
            self.bits,
            self.tactic,
        )
        if self.out_features != output.shape[1]:
            output = output[:, : self.out_features].contiguous()
        return output.reshape(*packed.shape, self.out_features)

    def forward_gelu(self, x, up=None, *, approximate="tanh"):
        if x.shape[-1] != self.in_features:
            raise ValueError("Wrong input feature count")
        return self.forward_packed(
            prepare_activation(
                x, self.bits, up=up, gelu=approximate, pack_reuse=self.pack_reuse
            )
        )

    def forward(self, x):
        return self.forward_packed(self.pack_input(x))


class IntegerProjectionGroup(torch.nn.Module):
    """Init-time packed concatenation; one GEMM per homogeneous sibling group."""

    def __init__(self, linears, *, contiguous_outputs=True):
        super().__init__()
        if len(linears) < 2 or any(not isinstance(m, IntegerLinear) for m in linears):
            raise ValueError("At least two integer projections are required")
        first = linears[0]
        if any(
            (
                m.in_features,
                m.bits,
                m.tactic,
                m.pack_reuse,
                m.biasless_epilogue,
                m.packed_weight.device,
            )
            != (
                first.in_features,
                first.bits,
                first.tactic,
                first.pack_reuse,
                first.biasless_epilogue,
                first.packed_weight.device,
            )
            for m in linears
        ):
            raise ValueError(
                "Projection format, input width, tactic and device must match"
            )
        self.sizes = tuple(m.out_features for m in linears)
        self.contiguous_outputs = contiguous_outputs
        physical = [m.packed_weight.shape[0] for m in linears]
        self.offsets = tuple(sum(physical[:i]) for i in range(len(linears)))
        bias = None
        if any(m.bias is not None for m in linears):
            bias = torch.cat(
                [
                    m.bias
                    if m.bias is not None
                    else torch.zeros(
                        m.packed_weight.shape[0],
                        device=m.packed_weight.device,
                        dtype=torch.bfloat16,
                    )
                    for m in linears
                ]
            )
        self.linear = IntegerLinear(
            torch.cat([m.packed_weight for m in linears]),
            torch.cat([m.weight_scale for m in linears]),
            bias,
            first.in_features,
            sum(physical),
            first.bits,
            first.tactic,
            first.pack_reuse,
            first.biasless_epilogue,
        )
        self._input = self._output = None
        self._remaining = set()

    def project(self, x, index):
        if self._input is not x or index not in self._remaining:
            self._input = x
            self._output = self.linear(x)
            self._remaining = set(range(len(self.sizes)))
        output = self._output.narrow(-1, self.offsets[index], self.sizes[index])
        if self.contiguous_outputs:
            output = output.contiguous()
        self._remaining.remove(index)
        if not self._remaining:
            self._input = self._output = None
        return output


class IntegerProjectionSlice(torch.nn.Module):
    def __init__(self, group, index):
        super().__init__()
        object.__setattr__(self, "_group", weakref.ref(group))
        self.index = index
        self.in_features = group.linear.in_features
        self.out_features = group.sizes[index]

    @property
    def weight(self):
        return self._group().linear.weight

    @property
    def bits(self):
        return self._group().linear.bits

    def forward(self, x):
        return self._group().project(x, self.index)
