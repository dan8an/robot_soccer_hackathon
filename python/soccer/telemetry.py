"""Rate-limited structured telemetry.

Per-frame printing at 12 Hz would itself slow the control loop, so routine
records are throttled while transitions and faults always print.
"""
import time

from .types import MotorCommand, WorldObservation


class Telemetry:
    def __init__(self, hz: float) -> None:
        self._min_interval = 1.0 / hz if hz > 0 else 0.0
        self._last_emit = 0.0
        self.frames = 0
        self.started = time.monotonic()

    def emit(
        self,
        state: str,
        obs: WorldObservation,
        command: MotorCommand,
        armed: bool,
        force: bool = False,
    ) -> None:
        self.frames += 1
        now = time.monotonic()
        if not force and (now - self._last_emit) < self._min_interval:
            return
        self._last_emit = now

        ball = obs.ball
        goal = obs.goal
        ball_text = (
            f"ball={ball.x:.2f},{ball.y:.2f}@{ball.confidence:.2f}"
            if ball.visible
            else f"ball=LOST({ball.lost_frames})"
        )
        goal_text = f" goal={goal.x:.2f}@{goal.confidence:.2f}" if goal.visible else ""
        # Which end we are aimed at is the difference between a goal and an
        # own goal, so it belongs in every line.
        side_text = f" facing={obs.facing_side[:4]}" if obs.facing_side != "UNKNOWN" else " facing=??"
        print(
            f"[{state:<7}] {ball_text}{goal_text}{side_text} "
            f"L={command.left:+.2f} R={command.right:+.2f} "
            f"({command.reason}) inf={obs.inference_ms:.0f}ms "
            f"{'ARMED' if armed else 'disarmed'}"
        )

    def transition(self, now_ms: float, old: str, new: str, reason: str) -> None:
        print(f"[STATE  ] {old} -> {new}  ({reason})")

    def event(self, message: str) -> None:
        print(f"[EVENT  ] {message}")

    def error(self, message: str) -> None:
        print(f"[ERROR  ] {message}")

    def rate_hz(self) -> float:
        elapsed = time.monotonic() - self.started
        return self.frames / elapsed if elapsed > 0 else 0.0
