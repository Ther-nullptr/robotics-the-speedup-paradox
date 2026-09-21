"""CUTLASS E4M3 Linear adapter for Thor and other Blackwell GPUs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from robotics_kernels.blackwell.extension_loader import (
    has_operator,
    load_extension,
    operator_namespace,
)
from robotics_kernels.blackwell.sampling_state import GreedyState
from robotics_kernels.blackwell.tensor_cache import tensor_cache_key


OP_NAMESPACE = "robotics_cutlass_fp8"
EXTENSION_ENV = "ROBOTICS_CUTLASS_FP8_SO"
DECODE_POLICY_ENV = "ROBOTICS_FP8_DECODE_POLICY"
SWAP_AB_ENV = "ROBOTICS_FP8_SWAP_AB"
PREFILL_SCALE_REFRESH_INTERVAL_ENV = "ROBOTICS_FP8_PREFILL_SCALE_REFRESH_INTERVAL"
PREFILL_MXFP8_ROLES_ENV = "ROBOTICS_FP8_PREFILL_MXFP8_ROLES"
PREFILL_MXFP8_ENABLED_ENV = "ROBOTICS_FP8_PREFILL_MXFP8_ENABLED"
SUPPORTED_PREFILL_MXFP8_ROLES = {"attn_qkv", "attn_o"}
PERIODIC_PREFILL_SCALE_ROLES = {
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_gate_up",
    "mlp_down",
}
LOW_CLOCK_THRESHOLD_ENV = "ROBOTICS_FP8_LOW_CLOCK_THRESHOLD_HZ"
GPU_DEVFREQ_ENV = "THOR_GPU_DEVFREQ_DIR"
DEFAULT_GPU_DEVFREQ_DIR = Path("/sys/class/devfreq/gpu-gpc-0")
DEFAULT_LOW_CLOCK_THRESHOLD_HZ = 954_000_000
DECODE_POLICY_IDS = {"balanced": 0, "low_clock": 1, "legacy": 2}
DEFAULT_EXTENSION_PATH = (
    Path(__file__).resolve().parents[3]
    / "build"
    / "robotics_cutlass_fp8"
    / "robotics_cutlass_fp8_ext.so"
)
REQUIRED_OPS = (
    "supports_device",
    "set_decode_policy",
    "get_decode_policy",
    "set_swap_ab_enabled",
    "get_swap_ab_enabled",
    "quantize_bf16",
    "add_rms_norm_quantize_bf16",
    "quantize_swiglu_bf16",
    "pack_linear_weight",
    "linear_forward",
    "linear_forward_producer",
    "linear_forward_packed",
    "linear_forward_packed_greedy",
    "linear_forward_packed_topk",
)
_LOAD_ATTEMPTED = False
_LOAD_ERROR: str | None = None


@dataclass(frozen=True)
class Fp8DecodePolicyStatus:
    policy: str
    policy_id: int
    requested: str
    max_freq_hz: int | None
    threshold_hz: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


_DECODE_POLICY_STATUS: Fp8DecodePolicyStatus | None = None
_SWAP_AB_ENABLED: bool | None = None


def resolve_fp8_prefill_scale_refresh_interval(
    value: str | int | None = None,
) -> int:
    """Resolve how often prefill activations refresh their dynamic FP8 scale."""

    raw = (
        (
            os.environ.get(PREFILL_SCALE_REFRESH_INTERVAL_ENV, "1")
            if value is None
            else str(value)
        )
        .strip()
        .lower()
    )
    if raw in {"static", "once", "never"}:
        return 2**31 - 1
    try:
        interval = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{PREFILL_SCALE_REFRESH_INTERVAL_ENV} must be a positive integer "
            "or 'static'"
        ) from exc
    if interval <= 0:
        raise ValueError(f"{PREFILL_SCALE_REFRESH_INTERVAL_ENV} must be positive")
    return interval


def resolve_fp8_prefill_mxfp8_roles(value: str | None = None) -> frozenset[str]:
    raw = (
        (os.environ.get(PREFILL_MXFP8_ROLES_ENV, "") if value is None else value)
        .strip()
        .lower()
    )
    if not raw or raw in {"none", "off"}:
        return frozenset()
    if raw in {"attention", "all"}:
        return frozenset(SUPPORTED_PREFILL_MXFP8_ROLES)
    roles = frozenset(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(roles - SUPPORTED_PREFILL_MXFP8_ROLES)
    if unknown:
        raise ValueError(
            f"{PREFILL_MXFP8_ROLES_ENV} contains unsupported roles: "
            + ", ".join(unknown)
        )
    return roles


def fp8_prefill_mxfp8_enabled() -> bool:
    return os.environ.get(PREFILL_MXFP8_ENABLED_ENV, "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class Fp8PrefillScaleState:
    """Reuse a per-module activation scale between multi-row prefill calls."""

    def __init__(
        self,
        interval: int | None = None,
        *,
        eligible: bool = True,
    ):
        self.eligible = bool(eligible)
        self.interval = (
            resolve_fp8_prefill_scale_refresh_interval(interval) if self.eligible else 1
        )
        self.scale: torch.Tensor | None = None
        self.prefill_calls = 0
        self.refreshes = 0
        self.reuses = 0
        self.operator_fallbacks = 0

    def configure(self, interval: int) -> None:
        self.interval = (
            resolve_fp8_prefill_scale_refresh_interval(interval) if self.eligible else 1
        )
        self.scale = None
        self.prefill_calls = 0
        self.refreshes = 0
        self.reuses = 0
        self.operator_fallbacks = 0

    @staticmethod
    def _rows(tensor: torch.Tensor, columns: int) -> int:
        return tensor.numel() // int(columns)

    def enabled_for(self, tensor: torch.Tensor, columns: int) -> bool:
        return self.interval > 1 and self._rows(tensor, columns) > 1

    def _refresh_required(self) -> bool:
        return self.scale is None or self.prefill_calls % self.interval == 0

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        namespace = _ops_namespace()
        if not self.enabled_for(tensor, tensor.shape[-1]):
            packed = namespace.quantize_bf16(tensor)
            return packed[0], packed[1]
        refresh = self._refresh_required()
        if refresh or not _has_op("quantize_bf16_with_scale"):
            packed = namespace.quantize_bf16(tensor)
            self.scale = packed[1]
            self.prefill_calls += 1
            self.refreshes += 1
            if not refresh:
                self.operator_fallbacks += 1
            return packed[0], packed[1]
        packed = namespace.quantize_bf16_with_scale(tensor, self.scale)
        self.prefill_calls += 1
        self.reuses += 1
        return packed, self.scale

    def quantize_swiglu(
        self,
        gate_up: torch.Tensor,
        intermediate_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        namespace = _ops_namespace()
        rows = self._rows(gate_up, 2 * int(intermediate_size))
        if self.interval <= 1 or rows <= 1:
            packed = namespace.quantize_swiglu_bf16(gate_up, int(intermediate_size))
            return packed[0], packed[1]
        refresh = self._refresh_required()
        if refresh or not _has_op("quantize_swiglu_bf16_with_scale"):
            packed = namespace.quantize_swiglu_bf16(gate_up, int(intermediate_size))
            self.scale = packed[1]
            self.prefill_calls += 1
            self.refreshes += 1
            if not refresh:
                self.operator_fallbacks += 1
            return packed[0], packed[1]
        packed = namespace.quantize_swiglu_bf16_with_scale(
            gate_up, int(intermediate_size), self.scale
        )
        self.prefill_calls += 1
        self.reuses += 1
        return packed, self.scale

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "eligible": self.eligible,
            "interval": self.interval,
            "prefill_calls": self.prefill_calls,
            "refreshes": self.refreshes,
            "reuses": self.reuses,
            "operator_fallbacks": self.operator_fallbacks,
        }


@dataclass
class Fp8BackendStatus:
    available: bool
    missing_ops: list[str]
    cuda_available: bool
    device_capability: int | None
    device_supported: bool | None
    namespace: str
    extension_path: str
    load_error: str | None
    decode_policy: dict[str, Any] | None
    swap_ab_enabled: bool | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Fp8GreedyState(GreedyState):
    """Compatibility name for shared device-resident sampling state."""

    input_error = "fused FP8 sampling requires CUDA batch size one"


_ops_namespace = partial(operator_namespace, OP_NAMESPACE)
_has_op = partial(has_operator, OP_NAMESPACE)


def resolve_fp8_decode_policy() -> Fp8DecodePolicyStatus:
    """Resolve the CUTLASS decode policy from an override or Thor clock."""

    requested = os.environ.get(DECODE_POLICY_ENV, "auto").strip().lower()
    aliases = {
        "low-clock": "low_clock",
        "lowclock": "low_clock",
        "default": "balanced",
    }
    requested = aliases.get(requested, requested)
    valid = {"auto", *DECODE_POLICY_IDS}
    if requested not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"{DECODE_POLICY_ENV} must be one of {choices}")

    try:
        threshold_hz = int(
            os.environ.get(
                LOW_CLOCK_THRESHOLD_ENV,
                str(DEFAULT_LOW_CLOCK_THRESHOLD_HZ),
            )
        )
    except ValueError as exc:
        raise ValueError(f"{LOW_CLOCK_THRESHOLD_ENV} must be an integer") from exc
    if threshold_hz <= 0:
        raise ValueError(f"{LOW_CLOCK_THRESHOLD_ENV} must be positive")

    devfreq_dir = Path(
        os.environ.get(GPU_DEVFREQ_ENV, str(DEFAULT_GPU_DEVFREQ_DIR))
    ).expanduser()
    max_freq_hz = None
    try:
        max_freq_hz = int((devfreq_dir / "max_freq").read_text().strip())
    except (OSError, ValueError):
        pass

    if requested == "auto":
        if max_freq_hz is not None and max_freq_hz <= threshold_hz:
            policy = "low_clock"
            source = "auto:max_freq"
        else:
            policy = "balanced"
            source = "auto:max_freq" if max_freq_hz is not None else "auto:fallback"
    else:
        policy = requested
        source = "environment"
    return Fp8DecodePolicyStatus(
        policy=policy,
        policy_id=DECODE_POLICY_IDS[policy],
        requested=requested,
        max_freq_hz=max_freq_hz,
        threshold_hz=threshold_hz,
        source=source,
    )


def configure_fp8_decode_policy() -> Fp8DecodePolicyStatus:
    global _DECODE_POLICY_STATUS
    status = resolve_fp8_decode_policy()
    namespace = _ops_namespace()
    if namespace is None or not _has_op("set_decode_policy"):
        raise RuntimeError("CUTLASS FP8 decode policy operator is unavailable")
    namespace.set_decode_policy(status.policy_id)
    _DECODE_POLICY_STATUS = status
    return status


def resolve_fp8_swap_ab_enabled() -> bool:
    value = os.environ.get(SWAP_AB_ENV, "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{SWAP_AB_ENV} must be a boolean value")


def configure_fp8_swap_ab() -> bool:
    global _SWAP_AB_ENABLED
    enabled = resolve_fp8_swap_ab_enabled()
    namespace = _ops_namespace()
    if namespace is None or not _has_op("set_swap_ab_enabled"):
        raise RuntimeError("CUTLASS FP8 Swap-AB operator is unavailable")
    namespace.set_swap_ab_enabled(enabled)
    _SWAP_AB_ENABLED = enabled
    return enabled


def _configure_loaded_extension() -> None:
    configure_fp8_decode_policy()
    configure_fp8_swap_ab()


def load_cutlass_fp8_extension(
    path: str | os.PathLike[str] | None = None,
) -> bool:
    """Load this backend while retaining its independent policy and error state."""
    global _LOAD_ATTEMPTED, _LOAD_ERROR
    loaded, _LOAD_ERROR = load_extension(
        OP_NAMESPACE,
        REQUIRED_OPS,
        lambda: [
            Path(path).expanduser()
            if path
            else Path(
                os.environ.get(EXTENSION_ENV, DEFAULT_EXTENSION_PATH)
            ).expanduser()
        ],
        configure=_configure_loaded_extension,
    )
    _LOAD_ATTEMPTED = True
    return loaded


def maybe_load_cutlass_fp8_extension() -> bool:
    global _LOAD_ERROR
    if all(_has_op(name) for name in REQUIRED_OPS):
        if _DECODE_POLICY_STATUS is None or _SWAP_AB_ENABLED is None:
            try:
                configure_fp8_decode_policy()
                configure_fp8_swap_ab()
            except (ValueError, RuntimeError) as exc:
                _LOAD_ERROR = str(exc)
                return False
        _LOAD_ERROR = None
        return True
    if _LOAD_ATTEMPTED:
        return False
    return load_cutlass_fp8_extension()


def fp8_backend_status(
    device: torch.device | int | None = None,
) -> Fp8BackendStatus:
    loaded = maybe_load_cutlass_fp8_extension()
    missing = [name for name in REQUIRED_OPS if not _has_op(name)]
    cc = None
    supported = None
    if torch.cuda.is_available():
        if device is None:
            index = torch.cuda.current_device()
        elif isinstance(device, int):
            index = device
        else:
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        major, minor = torch.cuda.get_device_capability(index)
        cc = major * 10 + minor
        if not missing:
            supported = bool(_ops_namespace().supports_device(cc))
    return Fp8BackendStatus(
        available=(
            loaded and not missing and torch.cuda.is_available() and bool(supported)
        ),
        missing_ops=missing,
        cuda_available=torch.cuda.is_available(),
        device_capability=cc,
        device_supported=supported,
        namespace=OP_NAMESPACE,
        extension_path=os.environ.get(EXTENSION_ENV, str(DEFAULT_EXTENSION_PATH)),
        load_error=_LOAD_ERROR,
        decode_policy=(
            _DECODE_POLICY_STATUS.to_dict()
            if _DECODE_POLICY_STATUS is not None
            else None
        ),
        swap_ab_enabled=_SWAP_AB_ENABLED,
    )


def require_fp8_backend(device: torch.device | int | None = None) -> None:
    status = fp8_backend_status(device)
    if not status.available:
        raise RuntimeError(f"CUTLASS FP8 backend is unavailable: {status.to_dict()}")


class Fp8ActivationQuantCache:
    """Reuse one dynamic E4M3 activation across sibling projections."""

    def __init__(self, fanout: int, *, periodic_scale_eligible: bool = False):
        self.fanout = int(fanout)
        if self.fanout <= 0:
            raise ValueError("activation quantization fanout must be positive")
        self._key: tuple[Any, ...] | None = None
        self._packed: tuple[torch.Tensor, torch.Tensor] | None = None
        self._remaining = 0
        self._prefill_scale_state = Fp8PrefillScaleState(
            eligible=periodic_scale_eligible
        )

    _tensor_key = staticmethod(tensor_cache_key)

    def get(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._tensor_key(tensor)
        if key != self._key or self._packed is None:
            packed = self._prefill_scale_state.quantize(tensor)
            self._key = key
            self._packed = packed
            self._remaining = self.fanout
        result = self._packed
        self._remaining -= 1
        if self._remaining == 0:
            self._key = None
            self._packed = None
        return result


class Fp8CutlassLinear(nn.Module):
    """Prequantized E4M3 weight, dynamic E4M3 activation, BF16 output."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        metadata: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        role: str,
        target_name: str,
        mxfp8_weight: torch.Tensor | None = None,
        mxfp8_weight_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("metadata", metadata)
        self.register_buffer(
            "mxfp8_weight",
            mxfp8_weight
            if mxfp8_weight is not None
            else torch.empty(
                0,
                dtype=torch.float8_e4m3fn,
                device=packed_weight.device,
            ),
        )
        self.register_buffer(
            "mxfp8_weight_scale",
            mxfp8_weight_scale
            if mxfp8_weight_scale is not None
            else torch.empty(0, dtype=torch.uint8, device=packed_weight.device),
        )
        self.register_buffer(
            "_weight_descriptor",
            torch.empty(0, dtype=torch.bfloat16, device=packed_weight.device),
            persistent=False,
        )
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16).contiguous())
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.fp8_role = role
        self.fp8_target_name = target_name
        self.fp8_backend = "cutlass_e4m3"
        self._mxfp8_prefill_calls = 0
        self.activation_quant_cache: Fp8ActivationQuantCache | None = None
        self._prefill_scale_state = Fp8PrefillScaleState(
            eligible=role in PERIODIC_PREFILL_SCALE_ROLES
        )
        self._fused_greedy_passthrough = False
        producer_enabled = os.environ.get("ROBOTICS_FP8_MIXED_INPUT_PRODUCER", "1")
        self._producer_enabled = producer_enabled.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._producer_min_m = int(os.environ.get("ROBOTICS_FP8_PRODUCER_MIN_M", "16"))
        self._producer_max_n = int(
            os.environ.get("ROBOTICS_FP8_PRODUCER_MAX_N", "1280")
        )
        self._producer_max_k = int(
            os.environ.get("ROBOTICS_FP8_PRODUCER_MAX_K", "5120")
        )
        prefill_wide = os.environ.get("ROBOTICS_FP8_PREFILL_WIDE", "0")
        self._prefill_wide_enabled = prefill_wide.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    @classmethod
    def from_linear(cls, linear: nn.Linear, role: str, target_name: str):
        if not linear.weight.is_cuda:
            raise RuntimeError("Fp8CutlassLinear requires CUDA weights")
        require_fp8_backend(linear.weight.device)
        weight = linear.weight.detach().to(torch.bfloat16).contiguous()
        packed = _ops_namespace().pack_linear_weight(weight, role, target_name)
        mxfp8_weight = None
        mxfp8_weight_scale = None
        if role in resolve_fp8_prefill_mxfp8_roles():
            from robotics_kernels.blackwell.mxfp8_linear import mxfp8_ops

            mxfp8_weight, mxfp8_weight_scale = mxfp8_ops().pack(weight)
        return cls(
            packed_weight=packed[0],
            weight_scale=packed[1],
            metadata=packed[2],
            bias=linear.bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
            role=role,
            target_name=target_name,
            mxfp8_weight=mxfp8_weight,
            mxfp8_weight_scale=mxfp8_weight_scale,
        )

    @property
    def weight(self):
        return self._weight_descriptor

    def pack_input(self, input_tensor: torch.Tensor):
        if input_tensor.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, "
                f"got {input_tensor.shape[-1]}"
            )
        return self._prefill_scale_state.quantize(input_tensor)

    def _should_use_mxfp8(self, input_tensor: torch.Tensor) -> bool:
        rows = input_tensor.numel() // self.in_features
        return (
            rows >= 128
            and self.mxfp8_weight.numel() > 0
            and fp8_prefill_mxfp8_enabled()
        )

    def forward_mxfp8(self, input_tensor: torch.Tensor) -> torch.Tensor:
        from robotics_kernels.blackwell.mxfp8_linear import mxfp8_ops

        self._mxfp8_prefill_calls += 1
        leading_shape = input_tensor.shape[:-1]
        matrix = input_tensor.reshape(-1, self.in_features).contiguous()
        packed, scales = mxfp8_ops().pack(matrix)
        tactic = 102
        output = mxfp8_ops().linear(
            packed,
            scales,
            self.mxfp8_weight,
            self.mxfp8_weight_scale,
            self.bias,
            tactic,
        )
        return output.reshape((*leading_shape, self.out_features))

    def forward_packed(
        self,
        packed_input: torch.Tensor,
        input_scale: torch.Tensor,
        leading_shape: tuple[int, ...] | torch.Size,
    ) -> torch.Tensor:
        rows = packed_input.numel() // self.in_features
        wide_prefill_roles = {
            "attn_qkv",
            "attn_o",
            "mlp_gate_up",
            "vision_attn_qkv",
            "vision_mlp_down",
            "lm_head",
        }
        if (
            self._prefill_wide_enabled
            and rows > 8
            and self.fp8_role in wide_prefill_roles
        ):
            output = _ops_namespace().linear_forward_packed_tactic(
                packed_input,
                input_scale,
                self.packed_weight,
                self.weight_scale,
                self.bias,
                6,
            )
        else:
            output = _ops_namespace().linear_forward_packed(
                packed_input,
                input_scale,
                self.packed_weight,
                self.weight_scale,
                self.bias,
            )
        return output.reshape((*leading_shape, self.out_features))

    def forward_producer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Convert BF16 A to E4M3 inside the CUTLASS mainloop producer."""

        input_tensor = input_tensor.to(torch.bfloat16).contiguous()
        return _ops_namespace().linear_forward_producer(
            input_tensor,
            self.packed_weight,
            self.weight_scale,
            self.bias,
        )

    def _should_use_producer(self, input_tensor: torch.Tensor) -> bool:
        if not self._producer_enabled:
            return False
        rows = input_tensor.numel() // self.in_features
        return (
            rows >= self._producer_min_m
            and self.out_features <= self._producer_max_n
            and self.in_features <= self._producer_max_k
        )

    def forward_swiglu(
        self,
        gate_up: torch.Tensor,
        intermediate_size: int,
        leading_shape: tuple[int, ...] | torch.Size,
    ) -> torch.Tensor:
        """Fuse SiLU, multiply, and dynamic E4M3 packing for a down GEMM."""

        packed, scale = self._prefill_scale_state.quantize_swiglu(
            gate_up.to(torch.bfloat16).contiguous(),
            int(intermediate_size),
        )
        return self.forward_packed(packed, scale, leading_shape)

    def set_fused_greedy_passthrough(self, enabled: bool) -> None:
        if enabled and self.fp8_role != "lm_head":
            raise RuntimeError("fused FP8 sampling requires an lm_head")
        self._fused_greedy_passthrough = bool(enabled)

    def forward_greedy(
        self, input_tensor: torch.Tensor, state: Fp8GreedyState
    ) -> torch.Tensor:
        """Run FP8 lm_head plus repetition/EOS processing and argmax."""

        if self.fp8_role != "lm_head":
            raise RuntimeError("fused FP8 sampling requires an lm_head")
        last_hidden = input_tensor.reshape(-1, self.in_features)[-1:].contiguous()
        packed_input, input_scale = self.pack_input(last_hidden)
        return _ops_namespace().linear_forward_packed_greedy(
            packed_input,
            input_scale,
            self.packed_weight,
            self.weight_scale,
            self.bias,
            state.seen_tokens,
            state.eos_tokens,
            state.generated_count,
            state.repetition_penalty,
            state.min_new_tokens,
        )

    def forward_topk(
        self,
        input_tensor: torch.Tensor,
        state: Fp8GreedyState,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the processed FP8 lm_head and return top-k scores/ids."""

        if self.fp8_role != "lm_head":
            raise RuntimeError("fused FP8 sampling requires an lm_head")
        last_hidden = input_tensor.reshape(-1, self.in_features)[-1:].contiguous()
        packed_input, input_scale = self.pack_input(last_hidden)
        result = _ops_namespace().linear_forward_packed_topk(
            packed_input,
            input_scale,
            self.packed_weight,
            self.weight_scale,
            self.bias,
            state.seen_tokens,
            state.eos_tokens,
            state.generated_count,
            state.repetition_penalty,
            state.min_new_tokens,
            int(top_k),
        )
        return result[0], result[1]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, "
                f"got {input_tensor.shape[-1]}"
            )
        if self._fused_greedy_passthrough:
            return input_tensor[..., -1:, :]
        if input_tensor.dtype != torch.bfloat16:
            input_tensor = input_tensor.to(torch.bfloat16)
        if self._should_use_mxfp8(input_tensor):
            return self.forward_mxfp8(input_tensor)
        if self.activation_quant_cache is not None:
            packed, scale = self.activation_quant_cache.get(input_tensor)
            return self.forward_packed(packed, scale, input_tensor.shape[:-1])
        if self._should_use_producer(input_tensor):
            return self.forward_producer(input_tensor)
        if not (
            self._prefill_wide_enabled
            or self._prefill_scale_state.enabled_for(input_tensor, self.in_features)
        ):
            return _ops_namespace().linear_forward(
                input_tensor,
                self.packed_weight,
                self.weight_scale,
                self.bias,
            )
        packed, scale = self.pack_input(input_tensor)
        return self.forward_packed(
            packed,
            scale,
            input_tensor.shape[:-1],
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"role={self.fp8_role!r}, target={self.fp8_target_name!r}, "
            f"prefill_mxfp8={self.mxfp8_weight.numel() > 0}"
        )


def _fp8_prefill_scale_states(model: nn.Module) -> list[Fp8PrefillScaleState]:
    states: dict[int, Fp8PrefillScaleState] = {}
    for module in model.modules():
        state = getattr(module, "_prefill_scale_state", None)
        if isinstance(state, Fp8PrefillScaleState) and state.eligible:
            states[id(state)] = state
        cache = getattr(module, "activation_quant_cache", None)
        cache_state = getattr(cache, "_prefill_scale_state", None)
        if isinstance(cache_state, Fp8PrefillScaleState) and cache_state.eligible:
            states[id(cache_state)] = cache_state
    return list(states.values())


def configure_fp8_prefill_scale_refresh(
    model: nn.Module,
    interval: str | int | None = None,
) -> dict[str, int]:
    """Set and reset periodic prefill scale state for an already loaded model."""

    states = _fp8_prefill_scale_states(model)
    if not states:
        return {"interval": 1, "states": 0}
    resolved = resolve_fp8_prefill_scale_refresh_interval(interval)
    for state in states:
        state.configure(resolved)
    return {"interval": resolved, "states": len(states)}


def fp8_prefill_scale_refresh_stats(model: nn.Module) -> dict[str, Any]:
    states = _fp8_prefill_scale_states(model)
    totals = {
        "states": len(states),
        "prefill_calls": 0,
        "refreshes": 0,
        "reuses": 0,
        "operator_fallbacks": 0,
    }
    intervals = set()
    for state in states:
        row = state.to_dict()
        intervals.add(row["interval"])
        for key in (
            "prefill_calls",
            "refreshes",
            "reuses",
            "operator_fallbacks",
        ):
            totals[key] += row[key]
    totals["interval"] = intervals.pop() if len(intervals) == 1 else None
    return totals


def reset_fp8_prefill_mxfp8_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, Fp8CutlassLinear):
            module._mxfp8_prefill_calls = 0


def fp8_prefill_mxfp8_stats(model: nn.Module) -> dict[str, Any]:
    modules = [
        module
        for module in model.modules()
        if isinstance(module, Fp8CutlassLinear) and module.mxfp8_weight.numel() > 0
    ]
    roles = sorted({module.fp8_role for module in modules})
    return {
        "modules": len(modules),
        "roles": roles,
        "calls": sum(module._mxfp8_prefill_calls for module in modules),
        "weight_bytes": sum(
            module.mxfp8_weight.numel() * module.mxfp8_weight.element_size()
            + module.mxfp8_weight_scale.numel()
            * module.mxfp8_weight_scale.element_size()
            for module in modules
        ),
        "enabled": fp8_prefill_mxfp8_enabled(),
    }
