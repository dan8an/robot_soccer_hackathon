"""Final gate between the controller and the motors."""
import math

from .config import Config
from .steering import clamp
from .types import MotorCommand


class SafetyLimiter:
    """Clamps, rejects non-finite values, and slew-limits normal commands.

    An emergency stop bypasses slew limiting entirely: when something is
    wrong the wheels go to zero this instant, not over a ramp.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._last_left = 0.0
        self._last_right = 0.0

    def emergency_stop(self, reason: str) -> MotorCommand:
        self._last_left = 0.0
        self._last_right = 0.0
        return MotorCommand.stop(reason)

    def limit(self, command: MotorCommand, dt_s: float) -> MotorCommand:
        cfg = self._config

        # A non-finite command is a bug upstream; refuse to pass it through.
        if not (math.isfinite(command.left) and math.isfinite(command.right)):
            return self.emergency_stop("non_finite_command")

        limit = cfg.max_motor_command
        left = clamp(command.left, -limit, limit)
        right = clamp(command.right, -limit, limit)

        if command.is_stop():
            return self.emergency_stop(command.reason)

        max_delta = cfg.motor_slew_per_second * max(dt_s, 1e-3)
        left = clamp(left, self._last_left - max_delta, self._last_left + max_delta)
        right = clamp(right, self._last_right - max_delta, self._last_right + max_delta)

        self._last_left = left
        self._last_right = right
        return MotorCommand(left=left, right=right, reason=command.reason)


class Watchdog:
    """Trips when the control loop stops feeding it."""

    def __init__(self, timeout_ms: float) -> None:
        self._timeout_ms = timeout_ms
        self._last_fed_ms: float | None = None

    def feed(self, now_ms: float) -> None:
        self._last_fed_ms = now_ms

    def expired(self, now_ms: float) -> bool:
        if self._last_fed_ms is None:
            return False
        return (now_ms - self._last_fed_ms) > self._timeout_ms
