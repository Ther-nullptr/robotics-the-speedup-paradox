"""The appendix tier ladder must select exact, cumulative module sets."""

import importlib
from pathlib import Path
import pytest


def api():
    assert (
        Path(__file__).parents[1] / "src/robotics_bench/optimizations/progressive.py"
    ).exists(), "Progressive quantization is missing"
    return importlib.import_module("robotics_bench.optimizations.progressive")


def test_cosmos_ladder_matches_appendix_counts_and_retains_other_sites():
    module = api()
    names = module.cosmos_candidate_sites()
    assert len(names) == 280
    previous = set()
    for tier, expected in enumerate((0, 8, 18, 28, 36, 46, 56, 84, 168, 224, 280)):
        mapping = module.cosmos_precision_map(names, tier)
        selected = {name for name, bits in mapping.items() if bits == 4}
        assert len(selected) == expected
        assert previous <= selected
        assert set(mapping.values()) <= {4, 8}
        previous = selected
    assert (
        module.cosmos_precision_map(names, 1)["blocks.20.cross_attn.output_proj"] == 4
    )
    assert (
        module.cosmos_precision_map(names, 1)["blocks.19.cross_attn.output_proj"] == 8
    )


def test_cosmos_ladder_rejects_unmatched_architecture_and_invalid_tiers():
    module = api()
    names = module.cosmos_candidate_sites()
    for tier in (-1, 11, True):
        with pytest.raises(ValueError):
            module.cosmos_precision_map(names, tier)
    with pytest.raises(ValueError, match="candidate"):
        module.cosmos_precision_map(names[:-1], 1)
    with pytest.raises(ValueError, match="candidate"):
        module.cosmos_precision_map(names + [names[0]], 1)
