"""Only complete native causal chunks may be omitted after the condition prefix."""

import pytest


def test_prefix_keeps_native_chunk_boundaries_and_full_latent_shape():
    from robotics_bench.optimizations.cosmos_prefix import prefix_plan

    plan = prefix_plan(41, 5, 4, 16)
    assert plan == {
        "input_frames": 41,
        "latent_frames": 11,
        "condition_frames": 5,
        "encoded_frames": 17,
        "factor": 4,
        "window": 16,
    }
    assert prefix_plan(33, 4, 4, 16)["encoded_frames"] == 17
    assert prefix_plan(41, 5, 4, 4)["encoded_frames"] == 17


@pytest.mark.parametrize(
    "values", [(40, 5, 4, 16), (41, 12, 4, 16), (41, 5, 4, 3), (41, 0, 4, 16)]
)
def test_invalid_prefix_geometry_is_rejected(values):
    from robotics_bench.optimizations.cosmos_prefix import prefix_plan

    with pytest.raises(ValueError):
        prefix_plan(*values)
