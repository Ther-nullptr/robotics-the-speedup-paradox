"""CPU-only optimization settings and direct measured comparisons."""

from dataclasses import dataclass
import math
import re
from statistics import median
from robotics_kernels.ampere_ada.tactics import TACTIC_IDS

PRECISION_SWITCHES = {
    "shared_quant",
    "activation_quant_fusion",
    "integer_grouped",
    "integer_pack_reuse",
    "integer_qkv",
    "integer_gate_up",
    "integer_group_views",
}

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
    "integer_grouped",
    "integer_pack_reuse",
    "integer_qkv",
    "integer_gate_up",
    "integer_group_views",
)


@dataclass(frozen=True)
class OptimizationConfig:
    switches: tuple[str, ...] = ()
    precision: str = "bf16"
    scopes: tuple[str, ...] = ("text",)
    tactic: int = 0
    quant_tier: int | None = None

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
        if type(self.tactic) is not int or self.tactic not in TACTIC_IDS:
            raise ValueError("Unknown integer tactic")
        if self.precision not in ("bf16", "int8", "int4", "fp8", "fp4"):
            raise ValueError("Unknown precision")
        if "integer_group_views" in self.enabled and not set(self.enabled) & {
            "integer_grouped",
            "integer_qkv",
            "integer_gate_up",
        }:
            raise ValueError(
                "integer_group_views requires an integer projection grouping switch"
            )
        if self.quant_tier is not None and (
            type(self.quant_tier) is not int
            or not 0 <= self.quant_tier <= 10
            or self.precision != "int8"
            or self.scopes != ("dit",)
        ):
            raise ValueError(
                "Progressive tiers require INT8 base precision and the Cosmos dit scope"
            )

    @property
    def precision_label(self):
        return f"w4-t{self.quant_tier}" if self.quant_tier else self.precision

    @property
    def enabled(self):
        return self.switches

    def to_dict(self):
        data = {
            "switches": list(self.enabled),
            "precision": self.precision,
            "scopes": list(self.scopes),
            "tactic": self.tactic,
        }
        if self.quant_tier is not None:
            data["quant_tier"] = self.quant_tier
        return data


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


def measurement_configurations(config, variants=None):
    """Build an explicit common-anchor cohort before loading the model."""
    rows = [("original", OptimizationConfig())]
    shared = tuple(s for s in config.enabled if s not in PRECISION_SWITCHES)
    if variants is not None or config.precision != "bf16":
        rows.append(
            (
                "optimized-bf16",
                OptimizationConfig(shared, "bf16", config.scopes, config.tactic),
            )
        )
    if variants is None:
        return rows + [("candidate", config)]
    if not isinstance(variants, list) or not variants:
        raise ValueError("Variants must be a nonempty list")
    used = {name for name, _ in rows}
    allowed = {"id", "switches", "precision", "scopes", "tactic", "quant_tier"}
    for data in variants:
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError("Unknown variant fields")
        name = data.get("id")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
            or name in used
        ):
            raise ValueError(
                "Variant IDs must be unique, safe filenames and not reference IDs"
            )
        current = OptimizationConfig(
            tuple(data.get("switches", config.enabled)),
            data.get("precision", config.precision),
            tuple(data.get("scopes", config.scopes)),
            data.get("tactic", config.tactic),
            data.get("quant_tier", config.quant_tier),
        )
        if {s for s in current.enabled if s not in PRECISION_SWITCHES} != set(shared):
            raise ValueError(
                "Variants must use the same shared optimization switches as their matched BF16 reference"
            )
        if set(
            current.enabled
        ) & PRECISION_SWITCHES and not current.precision.startswith("int"):
            raise ValueError("Integer switches require integer precision")
        used.add(name)
        rows.append((name, current))
    return rows


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
