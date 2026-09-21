"""A sweep shares a measured original anchor and explicit BF16 comparator."""

import pytest
from robotics_bench.optimizations import config as api


def test_explicit_shared_ablation_inserts_matching_bf16_reference():
    base = api.OptimizationConfig(("flow_loop", "cuda_graph"))
    rows = api.measurement_configurations(
        base,
        [
            {"id": "int4-before", "precision": "int4"},
            {
                "id": "int4-cache",
                "precision": "int4",
                "switches": [
                    "flow_loop",
                    "cuda_graph",
                    "condition_cache",
                    "condition_projection_cache",
                ],
            },
        ],
        allow_shared_variants=True,
    )
    candidate_index = next(
        i for i, (name, _) in enumerate(rows) if name == "int4-cache"
    )
    assert rows[candidate_index - 1][1].precision == "bf16"
    assert rows[candidate_index - 1][1].enabled == rows[candidate_index][1].enabled


def test_sweep_keeps_shared_bf16_and_distinct_integer_variants():
    assert hasattr(api, "measurement_configurations"), "Sweep configuration is missing"
    base = api.OptimizationConfig(("flow_loop", "cuda_graph"))
    rows = api.measurement_configurations(
        base,
        [
            {
                "id": "w8",
                "precision": "int8",
                "switches": ["flow_loop", "cuda_graph", "integer_grouped"],
            },
            {
                "id": "w4",
                "precision": "int4",
                "switches": ["flow_loop", "cuda_graph", "integer_pack_reuse"],
            },
        ],
    )
    assert [name for name, _ in rows] == ["original", "optimized-bf16", "w8", "w4"]
    assert rows[0][1].enabled == ()
    assert rows[1][1] == base
    assert rows[2][1].precision == "int8" and rows[3][1].precision == "int4"


@pytest.mark.parametrize("names", [["original"], ["a", "a"], ["../escape"]])
def test_sweep_rejects_duplicate_or_unsafe_ids(names):
    assert hasattr(api, "measurement_configurations"), "Sweep configuration is missing"
    with pytest.raises(ValueError):
        api.measurement_configurations(
            api.OptimizationConfig(), [{"id": n} for n in names]
        )


def test_variant_inherits_explicit_tier_and_cannot_change_shared_fusions():
    base = api.OptimizationConfig(precision="int8", scopes=("dit",), quant_tier=3)
    rows = api.measurement_configurations(base, [{"id": "tuned", "tactic": 1}])
    assert rows[-1][1].quant_tier == 3
    with pytest.raises(ValueError, match="shared"):
        api.measurement_configurations(
            api.OptimizationConfig(("flow_loop",)),
            [
                {
                    "id": "different",
                    "precision": "int8",
                    "switches": ["flow_loop", "norm"],
                },
            ],
        )
