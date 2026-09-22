"""Opt-in convolution patch preserving VAE causal padding and feature caches."""

from contextlib import contextmanager

from .patching import Patches


def selection_reason(weight_shape, stride, policy):
    """Return the reason a module is outside an explicitly selected family."""
    if policy not in ("all_supported", "c96"):
        raise ValueError("Unknown convolution selection policy")
    if policy == "c96" and (
        tuple(weight_shape) != (96, 96, 3, 3, 3) or tuple(stride) != (1, 1, 1)
    ):
        return "c96 policy selects only 96-to-96, 3x3x3, stride-one convolution"
    return None


@contextmanager
def optimize_convolutions(
    tokenizer, *, tactic=0, module_names=None, policy="all_supported"
):
    """Patch supported Conv2d/3d ``_conv_forward`` methods within one tokenizer.

    The original public forward method stays intact, including causal cache
    concatenation and asymmetric padding. Weights must remain immutable during
    the context. Unsupported modules are retained and recorded as native.
    ``module_names`` optionally limits dispatch to a measured allowlist.
    ``policy='c96'`` selects only 96-to-96, 3x3x3, stride-one modules and uses
    CUTLASS only for channels-last 3D inputs; other layouts retain native code.
    It is an opt-in family ablation, not an automatic fastest-kernel selector.
    """
    import torch
    from robotics_kernels.ampere_ada.convolution import PackedConvolution, TACTICS

    if tactic not in TACTICS:
        raise ValueError("Unknown convolution tactic")
    selection_reason((), (), policy)
    if tokenizer.training:
        raise ValueError("Convolution optimization is inference-only")
    allowed = None if module_names is None else set(module_names)
    patches = Patches()
    report = {
        "tactic": tactic,
        "policy": policy,
        "modules": {},
        "layout": "input and output conversion included",
        "numerics": "FP32 accumulation with bias before BF16 output rounding; approximate relative to native BF16 convolution followed by bias addition",
        "calls_boundary": "host dispatches; CUDA Graph replay calls are excluded",
    }
    try:
        for name, module in tokenizer.named_modules():
            if not isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                continue
            row = {
                "type": type(module).__name__,
                "weight_shape": list(module.weight.shape),
                "backend": "native",
                "calls": 0,
                "native_fallback_calls": 0,
                "native_fallback_reasons": {},
            }
            report["modules"][name] = row
            if allowed is not None and name not in allowed:
                row["reason"] = "outside explicit module allowlist"
                continue
            reason = selection_reason(module.weight.shape, module.stride, policy)
            if reason:
                row["reason"] = reason
                continue
            try:
                packed = PackedConvolution.from_module(module, tactic=tactic)
            except ValueError as exc:
                row["reason"] = str(exc)
                continue
            row["backend"] = packed.backend
            row["plan"] = packed.plan
            original_weight, original_bias = module.weight, module.bias
            original_forward = module._conv_forward

            def forward(
                module,
                x,
                weight,
                bias,
                packed=packed,
                row=row,
                original_weight=original_weight,
                original_bias=original_bias,
                original_forward=original_forward,
            ):
                if (
                    module.training
                    or weight is not original_weight
                    or bias is not original_bias
                ):
                    raise ValueError(
                        "Convolution module changed while its immutable weight pack was active"
                    )
                if policy == "c96" and not x.is_contiguous(
                    memory_format=torch.channels_last_3d
                ):
                    reason = "c96 policy requires channels_last_3d input"
                    row["native_fallback_calls"] += 1
                    reasons = row["native_fallback_reasons"]
                    reasons[reason] = reasons.get(reason, 0) + 1
                    return original_forward(x, weight, bias)
                row["calls"] += 1
                return packed(x)

            patches.bind(module, "_conv_forward", forward)
        if allowed is not None and allowed - set(report["modules"]):
            raise ValueError("Convolution module allowlist contains unknown names")
        yield report
    finally:
        patches.restore()
