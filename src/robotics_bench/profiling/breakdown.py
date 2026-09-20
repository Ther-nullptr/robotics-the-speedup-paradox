"""Measured CUDA work categories, kept separate from clean service timings."""

from collections import defaultdict
import math


def category(event):
    name = event.get("name", "").lower()
    if event.get("cat") in ("gpu_memcpy", "gpu_memset"):
        return "GPU transfers"
    if "cudnn" in name or "conv" in name:
        return "Convolution"
    if "gemm" in name or "gemv" in name:
        return "Matrix multiplication"
    if "flash" in name or "softmax" in name:
        return "Attention and softmax"
    if name.startswith("_prepare") or "quantize" in name:
        return "Quantization and packing"
    if name in ("_rope", "_residual", "_modulate", "_norm_affine", "_gelu_mul"):
        return "Owned pointwise fusion"
    if "norm" in name or "reduce_kernel" in name or "elementwise" in name:
        return "Other pointwise and reductions"
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
