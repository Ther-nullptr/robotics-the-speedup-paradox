"""Optional command holding for research rollouts on unchanged native ticks."""

import math


class CommandClock:
    def __init__(
        self,
        *,
        control_hz,
        native_tick_seconds,
        infer=None,
        tape=None,
        fixed_action=None,
        execute_horizon=4,
    ):
        if not math.isfinite(native_tick_seconds) or native_tick_seconds <= 0:
            raise ValueError("native_tick_seconds must be finite and positive")
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        ticks = 1 / (native_tick_seconds * control_hz)
        if round(ticks) < 1 or not math.isclose(ticks, round(ticks), rel_tol=1e-10):
            raise ValueError(
                "control_hz must correspond to an integer number of native ticks per command"
            )
        if sum(source is not None for source in (infer, tape, fixed_action)) != 1:
            raise ValueError("Supply exactly one command source")
        if type(execute_horizon) is not int or execute_horizon < 1:
            raise ValueError("execute_horizon must be a positive integer")
        self.control_hz = float(control_hz)
        self.hold_ticks = round(ticks)
        self.infer, self.tape, self.fixed_action = infer, tape, fixed_action
        self.execute_horizon = execute_horizon
        self.command_updates = self.inference_calls = self._next_tick = 0
        self._current = self._chunk = None

    def sample(self, tick, observation):
        """Return the current command; the caller still advances every native tick."""
        if tick != self._next_tick:
            raise ValueError(
                "CommandClock requires consecutive native ticks starting at zero"
            )
        if tick % self.hold_ticks == 0:
            if self.tape is not None:
                # The tape already has a native-time axis. Causal downsampling
                # preserves its duration; do not stretch every source sample.
                self._current = self.tape[tick]
            elif self.infer is not None:
                offset = self.command_updates % self.execute_horizon
                if offset == 0:
                    self._chunk = self.infer(observation)
                    if len(self._chunk) < self.execute_horizon:
                        raise ValueError(
                            "Policy returned fewer actions than execute_horizon"
                        )
                    self.inference_calls += 1
                self._current = self._chunk[offset]
            else:
                self._current = self.fixed_action
            self.command_updates += 1
        self._next_tick += 1
        return self._current
