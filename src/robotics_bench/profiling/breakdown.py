"""Measured CUDA work categories, kept separate from clean service timings."""

from collections import defaultdict
import math


def category(event):
    name = event.get("name", "").lower()
    if event.get("cat") in ("gpu_memcpy", "gpu_memset"):
        return "GPU transfers"
    if "splitkreduce" in name or "cublaslt::epilogue" in name:
        return "Matrix reduction and epilogue"
    if any(
        token in name for token in ("copy_kernel", "catarraybatchedcopy", "cast_kernel")
    ):
        return "Copies and dtype conversions"
    if "cudnn" in name or "conv" in name:
        return "Convolution"
    if "gemm" in name or "gemv" in name:
        if any(
            token in name
            for token in (
                "defaultgemmwithvisitor<signed char",
                "int4b_t",
                "integer_subbyte<4",
                "tensorop_s8",
            )
        ):
            return "Integer matrix multiplication"
        if any(
            token in name for token in ("bf16", "bfloat16", "float", "half", "gemvx")
        ):
            return "Floating matrix multiplication"
        return "Matrix multiplication"
    if "flash" in name or "softmax" in name or "fmha" in name:
        return "Attention and softmax"
    if name.startswith("_vae_norm_affine"):
        return "VAE pointwise fusion"
    if name.startswith("_prepare_norm_modulation"):
        return "Normalization modulation and quantization fusion"
    if name.startswith("_prepare_residual_norm_modulation"):
        return "Residual normalization modulation and quantization fusion"
    if name.startswith("_prepare_modulation"):
        return "Modulation and quantization fusion"
    if name.startswith("_prepare") or "quantize" in name:
        return "Quantization and packing"
    if name in ("_rope", "_residual", "_modulate", "_norm_affine", "_gelu_mul"):
        return "Owned pointwise fusion"
    if (
        any(token in name for token in ("layer_norm", "layernorm", "rmsnorm", "_norm_"))
        or "reduce_kernel" in name
        or "rsqrt" in name
        or "pow_tensor_scalar" in name
    ):
        return "Normalization and reductions"
    if "elementwise" in name:
        return "Other elementwise arithmetic"
    return "Unclassified GPU work"


def summarize(trace, name):
    values = defaultdict(float)
    for event in trace.get("traceEvents", []):
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        duration = event.get("dur")
        if (
            not isinstance(duration, (float, int))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("Invalid CUDA event duration")
        values[category(event)] += duration / 1000
    if not values:
        raise ValueError("No measured CUDA events in trace")
    return {
        "name": name,
        "total": sum(values.values()),
        "components": [
            {"name": name, "value": value} for name, value in values.items()
        ],
    }


def document(profiles, sources, title):
    return {
        "schema_version": 1,
        "title": title,
        "metric": {
            "name": "Sum of profiled GPU operation durations",
            "unit": "ms",
            "aggregation": "one diagnostic policy call; not wall-clock latency",
            "lower_is_better": True,
        },
        "profiles": profiles,
        "notes": [
            "Only CUDA kernel/memcpy/memset events are included; nested CPU operator durations are excluded.",
            "GPU work durations may overlap. These sums are diagnostic and do not replace clean policy-service timing.",
            "Names are coarse kernel-family categories; unclassified work is explicitly retained.",
        ],
        "sources": sources,
    }


def pages(data):
    """Skill breakdowns accept at most four rows; repeat the anchor per page."""
    profiles = data["profiles"]
    if not profiles:
        raise ValueError("A breakdown needs at least one profile")
    if len(profiles) <= 4:
        return [data]
    return [
        {
            **data,
            "title": data["title"] + f" (page {i // 3 + 1})",
            "profiles": [profiles[0], *profiles[i : i + 3]],
        }
        for i in range(1, len(profiles), 3)
    ]
