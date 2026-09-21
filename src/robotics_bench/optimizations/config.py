"""CPU-only optimization settings and direct measured comparisons."""

from dataclasses import dataclass
import math
from statistics import median

SWITCHES = (
    "flow_loop",
    "mask_cache",
    "condition_cache",
    "empty_image_cache",
    "rope",
    "gated_residual",
    "gelu_mul",
    "norm",
    "projection_fusion",
    "cuda_graph",
    "shared_quant",
    "modulation",
    "activation_quant_fusion",
)


@dataclass(frozen=True)
class OptimizationConfig:
    switches: tuple[str, ...] = ()
    precision: str = "bf16"
    scopes: tuple[str, ...] = ("text",)
    tactic: int = 0

    def __post_init__(self):
        if not isinstance(self.switches, tuple):
            raise ValueError("switches must be a tuple of names")
        if any(not isinstance(s, str) or s not in SWITCHES for s in self.switches):
            raise ValueError("Unknown optimization switch")
        if len(set(self.switches)) != len(self.switches):
            raise ValueError("Duplicate optimization switch")
        if not self.scopes or any(
            s not in ("text", "expert", "vision", "projector", "dit")
            for s in self.scopes
        ):
            raise ValueError("Unknown quantization scope")
        if type(self.tactic) is not int or self.tactic not in (0, 1):
            raise ValueError("Unknown integer tactic")
        if self.precision not in ("bf16", "int8", "int4", "fp8", "fp4"):
            raise ValueError("Unknown precision")

    @property
    def enabled(self):
        return self.switches

    def to_dict(self):
        return {
            "switches": list(self.enabled),
            "precision": self.precision,
            "scopes": list(self.scopes),
            "tactic": self.tactic,
        }


def _samples(values):
    values = list(values)
    if not values or any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v <= 0
        for v in values
    ):
        raise ValueError("Timing samples must be nonempty finite positive numbers")
    return values


def compare_samples(baseline, candidate, *, exact, max_abs):
    baseline, candidate = _samples(baseline), _samples(candidate)
    if type(exact) is not bool or not math.isfinite(max_abs) or max_abs < 0:
        raise ValueError("Invalid numerical comparison")
    if exact and max_abs != 0:
        raise ValueError("Exact comparison must have zero error")
    before, after = median(baseline), median(candidate)
    return {
        "statistic": "median",
        "baseline_samples_ms": baseline,
        "candidate_samples_ms": candidate,
        "baseline_ms": before,
        "candidate_ms": after,
        "speedup_vs_baseline": before / after,
        "saved_ms": before - after,
        "exact": exact,
        "max_abs": max_abs,
    }
