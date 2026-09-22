"""Time partition and analytical endpoint checks for residual proposals."""

import numpy as np
import pytest


def test_switch_partition_preserves_requested_delay_and_native_sample_times():
    from robotics_bench.kinetix.calibration import partition_tick

    plan = partition_tick(0.004, native_dt=1 / 60, frame_skip=2, factor=2)
    assert sum(plan["durations"]) == pytest.approx(1 / 30)
    assert sum(
        dt for dt, old in zip(plan["durations"], plan["old"], strict=True) if old
    ) == pytest.approx(0.004)
    assert plan["starts"][plan["reward_indices"][1]] == 1 / 60
    assert plan["physical_steps"] == 5
    zero = partition_tick(0, native_dt=1 / 60, frame_skip=2, factor=2)
    assert zero["physical_steps"] == 4
    assert zero["durations"][-1] == 0
    assert not any(zero["old"])


@pytest.mark.parametrize("delay", [-0.001, float("nan"), 0.04, 1e-50])
def test_delay_outside_one_native_tick_is_rejected(delay):
    from robotics_bench.kinetix.calibration import partition_tick

    with pytest.raises(ValueError, match="delay"):
        partition_tick(delay, native_dt=1 / 60, frame_skip=2, factor=2)


def test_two_endpoint_correction_removes_the_single_anchor_full_delay_bias():
    from robotics_bench.kinetix.calibration import calibrate_field

    h, a = 1 / 60, 10
    native_new = np.array([3 * a * h**2])
    fine_new = np.array([10 * a * (h / 2) ** 2])
    one, two = calibrate_field(
        np.zeros(1), native_new, np.zeros(1), fine_new, np.zeros(1), 1
    )
    assert one[0] == pytest.approx(a * h**2 / 2)
    np.testing.assert_array_equal(two, [0])
    # A switch at the native H boundary should recover the two native steps:
    # no acceleration in the first, then one force kick and drift in the second.
    _, middle = calibrate_field(
        np.zeros(1),
        native_new,
        np.zeros(1),
        fine_new,
        np.array([3 * a * (h / 2) ** 2]),
        0.5,
    )
    assert middle[0] == pytest.approx(a * h**2)


def test_motion_envelope_catches_hidden_intermediate_clipping():
    from robotics_bench.kinetix.calibration import motion_bounds

    bounds = motion_bounds([14.99, 2.5], [20, 0], 0, 900, 0, 1 / 30)
    assert bounds["position"] > 15
    safe = motion_bounds([2.5, 2.5], [0, 0], 0, 10, 0, 1 / 30)
    assert safe["position"] < 15
    assert safe["velocity"] < 100


def test_impulse_partition_keeps_grid_fixed_and_preserves_action_occupancy():
    from robotics_bench.kinetix.calibration import partition_tick

    plans = [
        partition_tick(
            delay, native_dt=1 / 60, frame_skip=2, factor=2, split_switch=False
        )
        for delay in (0.004, 1 / 120, 1 / 120 + 1e-8)
    ]
    assert all(plan["physical_steps"] == 4 for plan in plans)
    assert plans[0]["durations"] == plans[1]["durations"] == plans[2]["durations"]
    for plan, delay in zip(plans, (0.004, 1 / 120, 1 / 120 + 1e-8), strict=True):
        assert sum(
            dt * w
            for dt, w in zip(plan["durations"], plan["old_fractions"], strict=True)
        ) == pytest.approx(delay)
