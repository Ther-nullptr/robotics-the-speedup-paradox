"""Device action metadata can be checked without materializing host values."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/kinetix/resident_action_runner.py"
    )
    spec = importlib.util.spec_from_file_location("resident_action_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class MetadataOnly:
    shape = (8, 6)
    dtype = np.dtype("float32")

    def __array__(self, *args, **kwargs):
        raise AssertionError("Action values must stay on device")


def test_layout_validation_never_reads_action_values():
    module().validate_action_layout(MetadataOnly(), horizon=8, action_dim=6)


@pytest.mark.parametrize("shape,dtype", [((4, 6), "float32"), ((8, 6), "int32")])
def test_invalid_layout_is_rejected_before_physics_execution(shape, dtype):
    value = MetadataOnly()
    value.shape, value.dtype = shape, np.dtype(dtype)
    with pytest.raises(ValueError, match="action chunk"):
        module().validate_action_layout(value, horizon=8, action_dim=6)


def test_historical_metadata_normalizes_json_sequences_without_hiding_changes():
    matches = module().metadata_matches
    stored = {"source_file": "old", "static": {"screen_dim": [500, 500]}, "dt": 1 / 60}
    current = {"source_file": "new", "static": {"screen_dim": (500, 500)}, "dt": 1 / 60}
    assert matches(stored, current)
    current["dt"] = 1 / 120
    assert not matches(stored, current)
