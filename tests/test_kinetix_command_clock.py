"""Command holding must preserve the native simulation/termination clock."""

import pytest


def test_policy_observes_only_at_chunk_requests_and_holds_commands_between_updates():
    from robotics_bench.kinetix.command_clock import CommandClock

    observations = []

    def infer(observation):
        observations.append(observation)
        return [observation + offset for offset in range(4)]

    clock = CommandClock(control_hz=10, native_tick_seconds=1 / 30, infer=infer)
    actions = [clock.sample(tick, 100 + tick) for tick in range(14)]
    assert actions == [100] * 3 + [101] * 3 + [102] * 3 + [103] * 3 + [112] * 2
    assert observations == [100, 112]
    assert clock.command_updates == 5
    assert clock.inference_calls == 2


def test_tape_is_sampled_causally_without_stretching_its_native_time_axis():
    from robotics_bench.kinetix.command_clock import CommandClock

    tape = list(range(14))
    clock = CommandClock(control_hz=10, native_tick_seconds=1 / 30, tape=tape)
    actions = [clock.sample(tick, None) for tick in range(len(tape))]
    assert len(actions) == len(tape)
    assert actions == [0] * 3 + [3] * 3 + [6] * 3 + [9] * 3 + [12] * 2
    assert clock.inference_calls == 0


def test_native_rate_keeps_the_existing_four_command_policy_schedule():
    from robotics_bench.kinetix.command_clock import CommandClock

    clock = CommandClock(
        control_hz=30,
        native_tick_seconds=1 / 30,
        infer=lambda observation: [observation + offset for offset in range(4)],
    )
    assert [clock.sample(tick, tick) for tick in range(7)] == list(range(7))
    assert clock.inference_calls == 2


@pytest.mark.parametrize("rate", [0, float("nan"), 20])
def test_invalid_or_nonuniform_native_grid_rates_are_rejected(rate):
    from robotics_bench.kinetix.command_clock import CommandClock

    with pytest.raises(ValueError, match="control_hz"):
        CommandClock(control_hz=rate, native_tick_seconds=1 / 30, tape=[0])
