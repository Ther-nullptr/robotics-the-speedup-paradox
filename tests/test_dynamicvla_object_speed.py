"""Scale only the task target's initial linear velocity, without changing physics."""

import copy
import math

import pytest

from robotics_bench.dynamicvla_dom.object_motion import scale_object_speed


def fixture():
    return {
        "scene": {
            "object": {
                "init_state": {
                    "lin_vel": [0.2, -0.1, 0.0],
                    "ang_vel": [0.0, 0.0, 0.3],
                    "pos": [1, 2, 3],
                }
            },
            "container": {"init_state": {"lin_vel": [0.1, 0.0, 0.0]}},
        },
        "sim": {"dt": 0.04},
        "events": {"perturbation": {"force": 2}},
    }


def test_speed_scale_matches_reference_field_and_does_not_mutate():
    original = fixture()
    before = copy.deepcopy(original)
    unchanged, baseline = scale_object_speed(original, 1.0)
    assert unchanged == original and unchanged is not original
    scaled, record = scale_object_speed(original, 1.25)
    assert scaled["scene"]["object"]["init_state"]["lin_vel"] == pytest.approx(
        [0.25, -0.125, 0]
    )
    expected = copy.deepcopy(original)
    expected["scene"]["object"]["init_state"]["lin_vel"] = scaled["scene"]["object"][
        "init_state"
    ]["lin_vel"]
    assert scaled == expected
    assert original == before
    assert record["source_speed_mps"] == pytest.approx(math.hypot(0.2, 0.1))
    assert record["configured_speed_mps"] == pytest.approx(
        baseline["configured_speed_mps"] * 1.25
    )
    stopped, _ = scale_object_speed(original, 0)
    assert stopped["scene"]["object"]["init_state"]["lin_vel"] == [0, 0, 0]
    assert stopped["scene"]["object"]["init_state"]["ang_vel"] == [0, 0, 0.3]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_reject_invalid_scale(value):
    with pytest.raises(ValueError, match="speed scale"):
        scale_object_speed(fixture(), value)


def test_reject_missing_or_invalid_source_velocity():
    with pytest.raises(ValueError, match="lin_vel"):
        scale_object_speed({}, 1)
    config = fixture()
    config["scene"]["object"]["init_state"]["lin_vel"] = [0, 1]
    with pytest.raises(ValueError, match="three"):
        scale_object_speed(config, 1.25)
