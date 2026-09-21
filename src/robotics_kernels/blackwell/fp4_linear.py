"""CUTLASS-only FP4 Linear adapter contract.

This module deliberately does not use vLLM private FP4 quantization ops. The
runtime backend is expected to be a local CUTLASS extension registered under
`torch.ops.robotics_cutlass_fp4`.
"""

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


OP_NAMESPACE = "robotics_cutlass_fp4"
ACTIVATION_MODE_ENV = "ROBOTICS_FP4_ACTIVATION_MODE"
W4A16_PERSISTENT_PREFILL_ENV = "ROBOTICS_W4A16_PERSISTENT_PREFILL"
W4A16_PERSISTENT_MIN_ROWS_ENV = "ROBOTICS_W4A16_PERSISTENT_MIN_ROWS"
W4A16_PERSISTENT_ROLES_ENV = "ROBOTICS_W4A16_PERSISTENT_ROLES"
W4A16_DOUBLE_BUFFER_ENV = "ROBOTICS_W4A16_DOUBLE_BUFFER"
EXTENSION_ENV = "ROBOTICS_CUTLASS_FP4_SO"
IMPL_ENV = "ROBOTICS_CUTLASS_FP4_IMPL"
DECODE_POLICY_ENV = "ROBOTICS_FP4_DECODE_POLICY"
LOW_CLOCK_THRESHOLD_ENV = "ROBOTICS_FP4_LOW_CLOCK_THRESHOLD_HZ"
MID_CLOCK_THRESHOLD_ENV = "ROBOTICS_FP4_MID_CLOCK_THRESHOLD_HZ"
GPU_DEVFREQ_ENV = "THOR_GPU_DEVFREQ_DIR"
DEFAULT_GPU_DEVFREQ_DIR = Path("/sys/class/devfreq/gpu-gpc-0")
DEFAULT_LOW_CLOCK_THRESHOLD_HZ = 540_000_000
DEFAULT_MID_CLOCK_THRESHOLD_HZ = 1_152_000_000
DECODE_POLICY_IDS = {
    "balanced": 0,
    "low_clock": 1,
    "mid_clock": 2,
    "legacy": 3,
}
DEFAULT_EXTENSION_PATH = (
    Path(__file__).resolve().parents[3]
    / "build"
    / "robotics_cutlass_fp4"
    / "robotics_cutlass_fp4_ext.so"
)
REQUIRED_OPS = (
    "supports_device",
    "set_decode_policy",
    "get_decode_policy",
    "pack_linear_weight",
    "linear_forward",
    "pack_activation_gemm_bf16",
    "linear_forward_packed_gemm_bf16",
    "linear_forward_packed_gemm_bf16_tactic",
    "linear_forward_packed_greedy_gemm_bf16",
    "linear_forward_packed_topk_gemm_bf16",
    "silu_mul_bf16",
    "linear_forward_swiglu_gemm_bf16",
    "fused_rope_qk_bf16",
    "fused_vision_rope_qk_bf16",
    "rms_norm_bf16",
    "add_rms_norm_bf16",
    "fused_cache_update_rope_bf16",
    "fused_prefill_cache_update_rope_bf16",
)
IMPL_OPS = {
    "gemm_bf16": ("pack_linear_weight_gemm_bf16", "linear_forward_gemm_bf16"),
    "gemv_fp4out": ("pack_linear_weight", "linear_forward"),
}
_LOAD_ATTEMPTED = False
_LOAD_ERROR: str | None = None


@dataclass(frozen=True)
class Fp4DecodePolicyStatus:
    policy: str
    policy_id: int
    requested: str
    max_freq_hz: int | None
    low_threshold_hz: int
    mid_threshold_hz: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


_DECODE_POLICY_STATUS: Fp4DecodePolicyStatus | None = None


def _w4a16_persistent_role_selected(role: str, requested: str) -> bool:
    selectors = {item.strip().lower() for item in requested.split(",") if item.strip()}
    if not selectors:
        return False
    if "all" in selectors or role.lower() in selectors:
        return True
    normalized = role.lower()
    return (
        ("language" in selectors and not normalized.startswith("vision_"))
        or ("vision" in selectors and normalized.startswith("vision_"))
        or ("attention" in selectors and "attn" in normalized)
        or ("mlp" in selectors and "mlp" in normalized)
    )


@dataclass
class Fp4BackendStatus:
    available: bool
    missing_ops: list[str]
    cuda_available: bool
    device_capability: int | None
    device_supported: bool | None
    namespace: str
    note: str
    available_impls: list[str]
    preferred_impl: str | None
    extension_path: str | None = None
    load_error: str | None = None
    decode_policy: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "missing_ops": self.missing_ops,
            "cuda_available": self.cuda_available,
            "device_capability": self.device_capability,
            "device_supported": self.device_supported,
            "namespace": self.namespace,
            "note": self.note,
            "available_impls": self.available_impls,
            "preferred_impl": self.preferred_impl,
            "extension_path": self.extension_path,
            "load_error": self.load_error,
            "decode_policy": self.decode_policy,
        }


class Fp4GreedyState(GreedyState):
    """Compatibility name for shared device-resident sampling state."""


_ops_namespace = partial(operator_namespace, OP_NAMESPACE)
_has_op = partial(has_operator, OP_NAMESPACE)


def _ops_registered() -> bool:
    return all(_has_op(name) for name in REQUIRED_OPS)


def resolve_fp4_decode_policy() -> Fp4DecodePolicyStatus:
    """Resolve the CUTLASS W4A4 decode tactic family from the Thor clock."""

    requested = os.environ.get(DECODE_POLICY_ENV, "auto").strip().lower()
    aliases = {
        "low-clock": "low_clock",
        "lowclock": "low_clock",
        "mid-clock": "mid_clock",
        "midclock": "mid_clock",
        "default": "balanced",
    }
    requested = aliases.get(requested, requested)
    valid = {"auto", *DECODE_POLICY_IDS}
    if requested not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"{DECODE_POLICY_ENV} must be one of {choices}")

    def read_threshold(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    low_threshold_hz = read_threshold(
        LOW_CLOCK_THRESHOLD_ENV, DEFAULT_LOW_CLOCK_THRESHOLD_HZ
    )
    mid_threshold_hz = read_threshold(
        MID_CLOCK_THRESHOLD_ENV, DEFAULT_MID_CLOCK_THRESHOLD_HZ
    )
    if low_threshold_hz >= mid_threshold_hz:
        raise ValueError(
            f"{LOW_CLOCK_THRESHOLD_ENV} must be below {MID_CLOCK_THRESHOLD_ENV}"
        )

    devfreq_dir = Path(
        os.environ.get(GPU_DEVFREQ_ENV, str(DEFAULT_GPU_DEVFREQ_DIR))
    ).expanduser()
    max_freq_hz = None
    try:
        max_freq_hz = int((devfreq_dir / "max_freq").read_text().strip())
    except (OSError, ValueError):
        pass

    if requested == "auto":
        source = "auto:max_freq" if max_freq_hz is not None else "auto:fallback"
        if max_freq_hz is not None and max_freq_hz <= low_threshold_hz:
            policy = "low_clock"
        elif max_freq_hz is not None and max_freq_hz <= mid_threshold_hz:
            policy = "mid_clock"
        else:
            policy = "balanced"
    else:
        policy = requested
        source = "environment"
    return Fp4DecodePolicyStatus(
        policy=policy,
        policy_id=DECODE_POLICY_IDS[policy],
        requested=requested,
        max_freq_hz=max_freq_hz,
        low_threshold_hz=low_threshold_hz,
        mid_threshold_hz=mid_threshold_hz,
        source=source,
    )


def configure_fp4_decode_policy() -> Fp4DecodePolicyStatus:
    global _DECODE_POLICY_STATUS
    status = resolve_fp4_decode_policy()
    namespace = _ops_namespace()
    if namespace is None or not _has_op("set_decode_policy"):
        raise RuntimeError("CUTLASS FP4 decode policy operator is unavailable")
    namespace.set_decode_policy(status.policy_id)
    _DECODE_POLICY_STATUS = status
    return status


def _candidate_extension_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get(EXTENSION_ENV)
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.append(DEFAULT_EXTENSION_PATH)
    return paths


def load_cutlass_fp4_extension(
    path: str | os.PathLike[str] | None = None,
) -> bool:
    """Load this backend while retaining its independent policy and error state."""
    global _LOAD_ATTEMPTED, _LOAD_ERROR
    loaded, _LOAD_ERROR = load_extension(
        OP_NAMESPACE,
        REQUIRED_OPS,
        lambda: (
            [Path(path).expanduser()]
            if path is not None
            else _candidate_extension_paths()
        ),
        configure=configure_fp4_decode_policy,
        check_exists=True,
        missing_description="missing required operators",
    )
    _LOAD_ATTEMPTED = True
    return loaded


def maybe_load_cutlass_fp4_extension() -> bool:
    global _LOAD_ERROR
    if _ops_registered():
        if _DECODE_POLICY_STATUS is None:
            try:
                configure_fp4_decode_policy()
            except (ValueError, RuntimeError) as exc:
                _LOAD_ERROR = str(exc)
                return False
        return True
    if _LOAD_ATTEMPTED:
        return False
    return load_cutlass_fp4_extension()


def cuda_device_capability(device: torch.device | int | None = None) -> int | None:
    if not torch.cuda.is_available():
        return None
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
    major, minor = torch.cuda.get_device_capability(index)
    return major * 10 + minor


def fp4_backend_status(device: torch.device | int | None = None) -> Fp4BackendStatus:
    maybe_load_cutlass_fp4_extension()
    missing = [name for name in REQUIRED_OPS if not _has_op(name)]
    available_impls = [
        name for name, ops in IMPL_OPS.items() if all(_has_op(op) for op in ops)
    ]
    cc = cuda_device_capability(device)
    supported = None
    if not missing and cc is not None:
        try:
            supported = bool(_ops_namespace().supports_device(cc))
        except Exception:
            supported = False
    available = not missing and torch.cuda.is_available() and bool(supported)
    preferred_impl = None
    note = (
        "ready"
        if available
        else "missing CUTLASS FP4 extension or unsupported CUDA device"
    )
    if available:
        try:
            preferred_impl = select_fp4_impl()
        except Exception as exc:
            note = f"{note}; implementation selection failed: {exc}"
    return Fp4BackendStatus(
        available=available,
        missing_ops=missing,
        cuda_available=torch.cuda.is_available(),
        device_capability=cc,
        device_supported=supported,
        namespace=OP_NAMESPACE,
        note=note,
        available_impls=available_impls,
        preferred_impl=preferred_impl,
        extension_path=os.environ.get(EXTENSION_ENV) or str(DEFAULT_EXTENSION_PATH),
        load_error=_LOAD_ERROR,
        decode_policy=(
            _DECODE_POLICY_STATUS.to_dict()
            if _DECODE_POLICY_STATUS is not None
            else None
        ),
    )


def require_fp4_backend(device: torch.device | int | None = None) -> None:
    status = fp4_backend_status(device)
    if status.available:
        return
    raise RuntimeError(
        "CUTLASS FP4 Linear backend is unavailable: "
        f"{status.to_dict()}. Build/register a CUTLASS extension providing "
        f"`torch.ops.{OP_NAMESPACE}` FP4 Linear operators."
    )


def select_fp4_impl() -> str:
    requested = os.environ.get(IMPL_ENV, "gemm_bf16").lower()
    aliases = {
        "gemm": "gemm_bf16",
        "bf16": "gemm_bf16",
        "gemv": "gemv_fp4out",
        "fp4out": "gemv_fp4out",
    }
    requested = aliases.get(requested, requested)
    if requested not in IMPL_OPS:
        raise ValueError(
            f"{IMPL_ENV} must be one of {sorted(IMPL_OPS)}; got {requested!r}"
        )
    pack_op, forward_op = IMPL_OPS[requested]
    if _has_op(pack_op) and _has_op(forward_op):
        return requested
    if requested != "gemm_bf16":
        raise RuntimeError(
            f"requested CUTLASS FP4 impl {requested!r} is unavailable; "
            f"missing {pack_op} or {forward_op}"
        )
    return "gemv_fp4out"


def cutlass_silu_mul_bf16(
    gate_up: torch.Tensor, intermediate_size: int, interleaved: bool = False
) -> torch.Tensor:
    require_fp4_backend(gate_up.device)
    return _ops_namespace().silu_mul_bf16(
        gate_up, int(intermediate_size), bool(interleaved)
    )


class Fp4ActivationPackCache:
    """Share one W4A4 activation pack across sibling Q/K/V projections."""

    def __init__(self, fanout: int = 3):
        self.fanout = int(fanout)
        if self.fanout <= 0:
            raise ValueError("activation-pack cache fanout must be positive")
        self._key: tuple[Any, ...] | None = None
        self._packed: tuple[torch.Tensor, torch.Tensor] | None = None
        self._remaining = 0

    _tensor_key = staticmethod(tensor_cache_key)

    def get(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._tensor_key(input_tensor)
        if key != self._key or self._packed is None:
            packed_input, input_scale = _ops_namespace().pack_activation_gemm_bf16(
                input_tensor
            )
            self._key = key
            self._packed = (packed_input, input_scale)
            self._remaining = self.fanout
        result = self._packed
        self._remaining -= 1
        if self._remaining == 0:
            self._key = None
            self._packed = None
        return result


class Fp4CutlassLinear(nn.Module):
    """Experimental Linear module backed by a CUTLASS FP4 extension.

    The default `gemm_bf16` implementation supports both prefill and decode
    inputs and writes BF16 output for the existing hidden-state path.

    Expected extension ABI:
    - `supports_device(int cc) -> bool`
    - legacy `gemv_fp4out`: `pack_linear_weight` and `linear_forward`
    - preferred `gemm_bf16`: `pack_linear_weight_gemm_bf16` and
      `linear_forward_gemm_bf16`
    """

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
        impl: str,
        activation_mode: str = "fp4",
        mixed_weight_scale: torch.Tensor | None = None,
        w4a16_persistent_prefill: bool = False,
        w4a16_persistent_min_rows: int = 128,
        w4a16_double_buffer: bool = False,
        fallback_linear: nn.Linear | None = None,
    ):
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("metadata", metadata)
        self.register_buffer(
            "mixed_weight_scale",
            mixed_weight_scale
            if mixed_weight_scale is not None
            else torch.empty(0, dtype=torch.bfloat16, device=packed_weight.device),
            persistent=False,
        )
        self.register_buffer(
            "persistent_weight",
            torch.empty(0, dtype=torch.bfloat16, device=packed_weight.device),
            persistent=False,
        )
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.fp4_role = role
        self.fp4_target_name = target_name
        self.fp4_backend = "cutlass"
        self.fp4_impl = impl
        self.fp4_activation_mode = activation_mode
        self.w4a16_persistent_prefill = bool(w4a16_persistent_prefill)
        self.w4a16_persistent_min_rows = int(w4a16_persistent_min_rows)
        self.w4a16_double_buffer = bool(w4a16_double_buffer)
        self._w4a16_double_buffer_manager = None
        self.fallback_linear = fallback_linear
        self.activation_pack_cache: Fp4ActivationPackCache | None = None
        self._prepared_activation: (
            tuple[tuple[Any, ...], torch.Tensor, torch.Tensor] | None
        ) = None
        self._fused_greedy_passthrough = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, role: str, target_name: str):
        if not linear.weight.is_cuda:
            raise RuntimeError("Fp4CutlassLinear requires CUDA weights")
        require_fp4_backend(linear.weight.device)
        impl = select_fp4_impl()
        activation_mode = os.environ.get(ACTIVATION_MODE_ENV, "fp4").strip().lower()
        if activation_mode not in {"fp4", "fp8", "bf16"}:
            raise ValueError(f"{ACTIVATION_MODE_ENV} must be 'fp4', 'fp8', or 'bf16'")
        if activation_mode == "fp8":
            from robotics_kernels.blackwell.fp4_mx_linear import mx_ops

            impl = "mx_w4a8"
            packed = mx_ops().pack_mxfp4_weight(
                linear.weight.detach().to(torch.bfloat16).contiguous(),
                role,
                target_name,
            )
        else:
            pack_op, _ = IMPL_OPS[impl]
            packed = getattr(_ops_namespace(), pack_op)(
                linear.weight.detach().contiguous(),
                role,
                target_name,
            )
        packed_weight, weight_scale, metadata = packed
        mixed_weight_scale = None
        w4a16_persistent_prefill = False
        w4a16_persistent_min_rows = 128
        w4a16_double_buffer = False
        if activation_mode == "bf16":
            from robotics_kernels.blackwell.fp4_mixed_linear import mixed_ops

            mixed_weight_scale = mixed_ops().repack_weight_scales(
                weight_scale, metadata
            )
            persistent_value = (
                os.environ.get(W4A16_PERSISTENT_PREFILL_ENV, "0").strip().lower()
            )
            w4a16_persistent_prefill = persistent_value not in {
                "0",
                "false",
                "no",
                "off",
            }
            persistent_roles = os.environ.get(W4A16_PERSISTENT_ROLES_ENV, "all")
            w4a16_persistent_prefill = (
                w4a16_persistent_prefill
                and _w4a16_persistent_role_selected(role, persistent_roles)
            )
            double_buffer_value = (
                os.environ.get(W4A16_DOUBLE_BUFFER_ENV, "0").strip().lower()
            )
            w4a16_double_buffer = double_buffer_value not in {
                "0",
                "false",
                "no",
                "off",
            }
            w4a16_double_buffer = (
                w4a16_double_buffer
                and _w4a16_persistent_role_selected(role, persistent_roles)
            )
            if w4a16_persistent_prefill and w4a16_double_buffer:
                raise ValueError(
                    f"{W4A16_PERSISTENT_PREFILL_ENV} and "
                    f"{W4A16_DOUBLE_BUFFER_ENV} are mutually exclusive"
                )
            w4a16_persistent_min_rows = int(
                os.environ.get(W4A16_PERSISTENT_MIN_ROWS_ENV, "128")
            )
            if w4a16_persistent_min_rows < 1:
                raise ValueError(f"{W4A16_PERSISTENT_MIN_ROWS_ENV} must be positive")
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(
            packed_weight=packed_weight,
            weight_scale=weight_scale,
            metadata=metadata,
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
            role=role,
            target_name=target_name,
            impl=impl,
            activation_mode=activation_mode,
            mixed_weight_scale=mixed_weight_scale,
            w4a16_persistent_prefill=w4a16_persistent_prefill,
            w4a16_persistent_min_rows=w4a16_persistent_min_rows,
            w4a16_double_buffer=w4a16_double_buffer,
        )

    @property
    def weight(self):
        if self.fallback_linear is not None:
            return self.fallback_linear.weight
        raise AttributeError(
            "Fp4CutlassLinear stores packed FP4 weights; dense `.weight` is unavailable"
        )

    def materialize_w4a16_persistent_weight(self) -> torch.Tensor:
        if self.fp4_activation_mode != "bf16":
            raise RuntimeError("persistent transformed weights require W4A16")
        if self.persistent_weight.numel() == 0:
            from robotics_kernels.blackwell.fp4_mixed_linear import mixed_ops

            self.persistent_weight = mixed_ops().materialize_weight_bf16(
                self.packed_weight,
                self.mixed_weight_scale,
                self.metadata,
            )
        return self.persistent_weight

    def clear_w4a16_persistent_weight(self) -> None:
        self.persistent_weight = torch.empty(
            0, dtype=torch.bfloat16, device=self.packed_weight.device
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, got {input_tensor.shape[-1]}"
            )
        if self.fp4_activation_mode == "bf16":
            from robotics_kernels.blackwell.fp4_mixed_linear import mixed_ops

            bf16_input = input_tensor.to(torch.bfloat16).contiguous()
            rows = bf16_input.numel() // self.in_features
            if self.w4a16_persistent_prefill and rows >= self.w4a16_persistent_min_rows:
                return torch.nn.functional.linear(
                    bf16_input,
                    self.materialize_w4a16_persistent_weight(),
                    self.bias,
                )
            manager = self._w4a16_double_buffer_manager
            if (
                self.w4a16_double_buffer
                and manager is not None
                and rows >= self.w4a16_persistent_min_rows
            ):
                execution_weight = manager.acquire(self)
                if execution_weight is not None:
                    output = torch.nn.functional.linear(
                        bf16_input, execution_weight, self.bias
                    )
                    manager.mark_consumed(self)
                    return output
            return mixed_ops().linear_w4a16(
                bf16_input,
                self.packed_weight,
                self.mixed_weight_scale,
                self.metadata,
                self.bias,
            )
        if self.fp4_activation_mode == "fp8":
            from robotics_kernels.blackwell.fp4_mx_linear import mx_ops

            return mx_ops().linear_w4a8(
                input_tensor.to(torch.bfloat16).contiguous(),
                self.packed_weight,
                self.weight_scale,
                self.metadata,
                self.bias,
            )
        if self._fused_greedy_passthrough:
            return input_tensor[..., -1:, :]
        prepared = self._prepared_activation
        self._prepared_activation = None
        if prepared is not None and prepared[0] == Fp4ActivationPackCache._tensor_key(
            input_tensor
        ):
            return self.forward_packed(
                prepared[1], prepared[2], input_tensor.shape[:-1]
            )
        if self.fp4_impl == "gemm_bf16" and self.activation_pack_cache is not None:
            packed_input, input_scale = self.activation_pack_cache.get(input_tensor)
            return self.forward_packed(
                packed_input, input_scale, input_tensor.shape[:-1]
            )
        _, forward_op = IMPL_OPS[self.fp4_impl]
        return getattr(_ops_namespace(), forward_op)(
            input_tensor,
            self.packed_weight,
            self.weight_scale,
            self.metadata,
            self.bias,
            self.fp4_role,
            self.fp4_target_name,
        )

    def prepare_packed_activation(
        self,
        input_tensor: torch.Tensor,
        packed_input: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> None:
        if self.fp4_activation_mode != "fp4":
            return
        if self.fp4_impl != "gemm_bf16":
            raise RuntimeError("prepared FP4 activations require gemm_bf16")
        self._prepared_activation = (
            Fp4ActivationPackCache._tensor_key(input_tensor),
            packed_input,
            input_scale,
        )

    def set_fused_greedy_passthrough(self, enabled: bool) -> None:
        if enabled and self.fp4_activation_mode != "fp4":
            raise RuntimeError("fused sampler is available only for W4A4")
        if enabled and (self.fp4_role != "lm_head" or self.fp4_impl != "gemm_bf16"):
            raise RuntimeError("fused greedy sampling requires a gemm_bf16 lm_head")
        self._fused_greedy_passthrough = bool(enabled)

    def forward_greedy(
        self, input_tensor: torch.Tensor, state: Fp4GreedyState
    ) -> torch.Tensor:
        """Run W4A4 lm_head plus repetition/EOS processing and argmax."""

        if (
            self.fp4_activation_mode != "fp4"
            or self.fp4_role != "lm_head"
            or self.fp4_impl != "gemm_bf16"
        ):
            raise RuntimeError("fused greedy sampling requires a gemm_bf16 lm_head")
        if input_tensor.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, got {input_tensor.shape[-1]}"
            )
        last_hidden = input_tensor.reshape(-1, self.in_features)[-1:].contiguous()
        packed_input, input_scale = self.pack_input(last_hidden)
        return _ops_namespace().linear_forward_packed_greedy_gemm_bf16(
            packed_input,
            input_scale,
            self.packed_weight,
            self.weight_scale,
            self.metadata,
            self.bias,
            state.seen_tokens,
            state.eos_tokens,
            state.generated_count,
            state.repetition_penalty,
            state.min_new_tokens,
            self.fp4_role,
            self.fp4_target_name,
        )

    def forward_topk(
        self,
        input_tensor: torch.Tensor,
        state: Fp4GreedyState,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the processed W4A4 lm_head and return only top-k scores/ids."""

        if (
            self.fp4_activation_mode != "fp4"
            or self.fp4_role != "lm_head"
            or self.fp4_impl != "gemm_bf16"
        ):
            raise RuntimeError("fused top-k sampling requires a gemm_bf16 lm_head")
        last_hidden = input_tensor.reshape(-1, self.in_features)[-1:].contiguous()
        packed_input, input_scale = self.pack_input(last_hidden)
        result = _ops_namespace().linear_forward_packed_topk_gemm_bf16(
            packed_input,
            input_scale,
            self.packed_weight,
            self.weight_scale,
            self.metadata,
            self.bias,
            state.seen_tokens,
            state.eos_tokens,
            state.generated_count,
            state.repetition_penalty,
            state.min_new_tokens,
            int(top_k),
            self.fp4_role,
            self.fp4_target_name,
        )
        return result[0], result[1]

    def pack_input(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fp4_activation_mode != "fp4":
            raise RuntimeError("packed-input reuse is available only for W4A4")
        if self.fp4_impl != "gemm_bf16":
            raise RuntimeError(
                "packed-input reuse requires the gemm_bf16 implementation"
            )
        if input_tensor.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, got {input_tensor.shape[-1]}"
            )
        packed = _ops_namespace().pack_activation_gemm_bf16(input_tensor)
        return packed[0], packed[1]

    def forward_packed(
        self,
        packed_input: torch.Tensor,
        input_scale: torch.Tensor,
        leading_shape: tuple[int, ...] | torch.Size,
    ) -> torch.Tensor:
        if self.fp4_activation_mode != "fp4":
            raise RuntimeError("packed-input forward is available only for W4A4")
        if self.fp4_impl != "gemm_bf16":
            raise RuntimeError(
                "packed-input forward requires the gemm_bf16 implementation"
            )
        output = _ops_namespace().linear_forward_packed_gemm_bf16(
            packed_input,
            input_scale,
            self.packed_weight,
            self.weight_scale,
            self.metadata,
            self.bias,
            self.fp4_role,
            self.fp4_target_name,
        )
        return output.reshape((*leading_shape, self.out_features))

    def forward_swiglu(
        self,
        gate_up: torch.Tensor,
        intermediate_size: int,
        interleaved: bool = False,
    ) -> torch.Tensor:
        """Fuse SwiGLU, activation packing, and this FP4 down projection."""

        if intermediate_size != self.in_features:
            raise ValueError(
                f"expected intermediate_size {self.in_features}, "
                f"got {intermediate_size}"
            )
        if self.fp4_activation_mode == "fp8":
            from robotics_kernels.blackwell.fp4_mx_linear import mx_ops

            packed, scale = mx_ops().pack_swiglu_mxfp8(
                gate_up.contiguous(),
                int(intermediate_size),
                bool(interleaved),
            )
            return mx_ops().linear_w4a8_packed(
                packed,
                scale,
                self.packed_weight,
                self.weight_scale,
                self.metadata,
                self.bias,
            )
        if self.fp4_activation_mode == "bf16" or self.fp4_impl != "gemm_bf16":
            return self.forward(
                cutlass_silu_mul_bf16(gate_up, intermediate_size, interleaved)
            )
        return _ops_namespace().linear_forward_swiglu_gemm_bf16(
            gate_up,
            int(intermediate_size),
            self.packed_weight,
            self.weight_scale,
            self.metadata,
            self.bias,
            self.fp4_role,
            self.fp4_target_name,
            bool(interleaved),
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"role={self.fp4_role!r}, target={self.fp4_target_name!r}, "
            f"impl={self.fp4_impl!r}, activation={self.fp4_activation_mode!r}, "
            f"persistent_prefill={self.w4a16_persistent_prefill}, "
            f"double_buffer={self.w4a16_double_buffer}, "
            f"dense_fallback={self.fallback_linear is not None}"
        )


@dataclass
class _W4A16BufferSlot:
    buffer: torch.Tensor
    owner: int | None = None
    view: torch.Tensor | None = None
    ready: torch.cuda.Event | None = None
    consumed: torch.cuda.Event | None = None


class W4A16DoubleBufferManager:
    """Learn a streaming call order and prefetch W4A16 weights into two slots."""

    def __init__(self, modules: list[Fp4CutlassLinear]):
        if len(modules) < 2:
            raise ValueError("W4A16 double buffering requires at least two modules")
        self.modules = list(modules)
        self.module_ids = {id(module) for module in modules}
        self.device = modules[0].packed_weight.device
        if any(module.packed_weight.device != self.device for module in modules):
            raise ValueError("W4A16 double-buffer modules must share one device")
        self.producer_stream = torch.cuda.Stream(device=self.device)
        self.order: list[Fp4CutlassLinear] = []
        self.recorded_ids: set[int] = set()
        self.order_index: dict[int, int] = {}
        self.slots: list[_W4A16BufferSlot] = []
        self.owner_to_slot: dict[int, int] = {}
        self.active_slots: dict[int, int] = {}
        self.finalized = False
        self.materializations = 0
        self.hits = 0
        self.misses = 0
        self.inline_recording_calls = 0
        self.sequence_misses = 0
        for module in modules:
            module._w4a16_double_buffer_manager = self

    def _finalize(self) -> None:
        max_elements = max(
            module.out_features * module.in_features for module in self.order
        )
        self.slots = [
            _W4A16BufferSlot(
                torch.empty(max_elements, dtype=torch.bfloat16, device=self.device)
            )
            for _ in range(2)
        ]
        self.order_index = {
            id(module): index for index, module in enumerate(self.order)
        }
        self.finalized = True

    @staticmethod
    def _weight_elements(module: Fp4CutlassLinear) -> int:
        return module.out_features * module.in_features

    def _schedule(self, module: Fp4CutlassLinear, slot_index: int) -> None:
        from robotics_kernels.blackwell.fp4_mixed_linear import mixed_ops

        slot = self.slots[slot_index]
        if slot.owner is not None:
            self.owner_to_slot.pop(slot.owner, None)
        elements = self._weight_elements(module)
        view = slot.buffer[:elements].view(module.out_features, module.in_features)
        with torch.cuda.stream(self.producer_stream):
            if slot.consumed is not None:
                self.producer_stream.wait_event(slot.consumed)
            mixed_ops().materialize_weight_bf16_out(
                module.packed_weight,
                module.mixed_weight_scale,
                module.metadata,
                view,
            )
            if slot.ready is None:
                slot.ready = torch.cuda.Event()
            slot.ready.record(self.producer_stream)
        owner = id(module)
        slot.owner = owner
        slot.view = view
        self.owner_to_slot[owner] = slot_index
        self.materializations += 1

    def acquire(self, module: Fp4CutlassLinear) -> torch.Tensor | None:
        module_id = id(module)
        if module_id not in self.module_ids:
            self.sequence_misses += 1
            return None
        if not self.finalized:
            if module_id not in self.recorded_ids:
                self.order.append(module)
                self.recorded_ids.add(module_id)
                self.inline_recording_calls += 1
                return None
            if module_id != id(self.order[0]) or len(self.order) < 2:
                self.inline_recording_calls += 1
                return None
            self._finalize()

        index = self.order_index.get(module_id)
        if index is None:
            self.sequence_misses += 1
            return None
        slot_index = self.owner_to_slot.get(module_id)
        if slot_index is None:
            slot_index = index & 1
            self._schedule(module, slot_index)
            self.misses += 1
        else:
            self.hits += 1

        slot = self.slots[slot_index]
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(slot.ready)

        next_module = self.order[(index + 1) % len(self.order)]
        next_id = id(next_module)
        next_slot_index = 1 - slot_index
        if self.owner_to_slot.get(next_id) != next_slot_index:
            self._schedule(next_module, next_slot_index)

        self.active_slots[module_id] = slot_index
        return slot.view

    def mark_consumed(self, module: Fp4CutlassLinear) -> None:
        module_id = id(module)
        slot_index = self.active_slots.pop(module_id, None)
        if slot_index is None:
            return
        slot = self.slots[slot_index]
        if slot.consumed is None:
            slot.consumed = torch.cuda.Event()
        slot.consumed.record(torch.cuda.current_stream(self.device))

    def status(self) -> dict[str, Any]:
        slot_bytes = sum(
            slot.buffer.numel() * slot.buffer.element_size() for slot in self.slots
        )
        return {
            "finalized": self.finalized,
            "sequence_modules": len(self.order),
            "slot_bytes": slot_bytes,
            "materializations": self.materializations,
            "hits": self.hits,
            "misses": self.misses,
            "inline_recording_calls": self.inline_recording_calls,
            "sequence_misses": self.sequence_misses,
        }

    def clear(self) -> dict[str, Any]:
        if self.slots:
            torch.cuda.synchronize(self.device)
        status = self.status()
        for module in self.modules:
            if module._w4a16_double_buffer_manager is self:
                module._w4a16_double_buffer_manager = None
        self.slots.clear()
        self.owner_to_slot.clear()
        self.active_slots.clear()
        return status
