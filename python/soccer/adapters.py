"""Hardware boundaries. Everything device-specific lives behind these."""
import time

from .config import Config
from .steering import clamp
from .types import Detection, MotorCommand, TrackedTarget, WorldObservation


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0


class MotorAdapter:
    """Maps a normalized differential command onto the mecanum chassis.

    From velocityController(angle=0) in sketch.ino the four wheels collapse
    into two virtual sides: m0/m3 are the right side, m1/m2 the left. Every
    command carries the firmware auto-stop duration, which is the hardware
    watchdog: if this process dies mid-drive the wheels still stop.
    """

    def __init__(self, config: Config, bridge) -> None:
        self._config = config
        self._bridge = bridge
        self._last_sent = (0.0, 0.0)

    def apply(self, command: MotorCommand) -> None:
        limit = self._config.max_motor_command
        left = clamp(command.left, -limit, limit)
        right = clamp(command.right, -limit, limit)

        if left == 0.0 and right == 0.0:
            self.stop(command.reason)
            return

        # Below the stiction floor the wheels do not turn at all, so a small
        # command silently does nothing. Scale both sides by the same factor:
        # scaling them independently would change the turn ratio.
        peak = max(abs(left), abs(right))
        floor = self._config.min_effective_speed
        if 0.0 < peak < floor:
            scale = floor / peak
            left *= scale
            right *= scale
            peak = floor
        # Scaling can push the pair past the limit; rescale to fit.
        if peak > limit:
            left *= limit / peak
            right *= limit / peak

        m_left = int(round(left * 255))
        m_right = int(round(right * 255))
        self._last_sent = (left, right)
        # m0, m3 = right side; m1, m2 = left side.
        self._bridge.call(
            "drive_raw",
            m_right,
            m_left,
            m_left,
            m_right,
            int(self._config.command_watchdog_ms),
        )

    def stop(self, reason: str = "stop") -> None:
        self._last_sent = (0.0, 0.0)
        self._bridge.call("drive_raw", 0, 0, 0, 0, 0)


class VisionAdapter:
    """Camera + model + trackers -> one WorldObservation per call."""

    def __init__(
        self, config: Config, camera, detector, trackers: dict, wall_detector=None
    ) -> None:
        self._config = config
        self._camera = camera
        self._detector = detector
        self._trackers = trackers
        self._wall = wall_detector
        self._frame_id = 0
        self._last_seq = -1
        self._last_observation: WorldObservation | None = None
        self.team_is_blue = False
        self.new_frames = 0
        self.repeats = 0

    def observe(self) -> WorldObservation:
        cfg = self._config
        now = monotonic_ms()

        frame, age_s, seq = self._camera.latest_frame()
        age_ms = age_s * 1000.0

        if frame is None or age_ms > cfg.vision_stale_ms:
            reason = "no_frame" if frame is None else f"stale_frame_{age_ms:.0f}ms"
            self._frame_id += 1
            # Age the trackers so lost-streaks keep advancing while blind.
            return WorldObservation(
                timestamp_ms=now,
                frame_id=self._frame_id,
                healthy=False,
                health_reason=reason,
                ball=self._trackers["ball"].update([], now),
                goal=self._trackers["goal"].update([], now),
                opponent=self._trackers["opponent"].update([], now),
            )

        # Same frame as last time: the camera is slower than the control loop.
        # Re-running the model would inflate detection streaks, so reuse the
        # previous result and only advance its age.
        if seq == self._last_seq and self._last_observation is not None:
            self.repeats += 1
            previous = self._last_observation
            return WorldObservation(
                timestamp_ms=now,
                frame_id=previous.frame_id,
                inference_ms=0.0,
                healthy=True,
                ball=self._aged(previous.ball, age_ms),
                goal=self._aged(previous.goal, age_ms),
                opponent=self._aged(previous.opponent, age_ms),
                facing_side=previous.facing_side,
                team_is_blue=self.team_is_blue,
            )

        self._last_seq = seq
        self._frame_id += 1
        self.new_frames += 1

        if self._wall is not None:
            self._wall.update(frame)

        started = time.monotonic()
        raw = self._detector.detect(frame, min_confidence=0.0)
        inference_ms = (time.monotonic() - started) * 1000.0

        detections = [
            Detection(label=d.label, confidence=d.confidence, x=d.x, y=d.y) for d in raw
        ]
        now = monotonic_ms()
        observed = WorldObservation(
            timestamp_ms=now,
            frame_id=self._frame_id,
            inference_ms=inference_ms,
            healthy=True,
            ball=self._trackers["ball"].update(detections, now),
            goal=self._trackers["goal"].update(detections, now),
            opponent=self._trackers["opponent"].update(detections, now),
            facing_side=self._wall.side.value if self._wall is not None else "UNKNOWN",
            team_is_blue=self.team_is_blue,
        )
        self._last_observation = observed
        return observed

    @staticmethod
    def _aged(target: TrackedTarget, age_ms: float) -> TrackedTarget:
        """Same target, current age. Streaks deliberately do not advance."""
        return TrackedTarget(
            visible=target.visible,
            confirmed=target.confirmed,
            confidence=target.confidence,
            x=target.x,
            y=target.y,
            age_ms=age_ms if target.visible else target.age_ms + age_ms,
            detected_frames=target.detected_frames,
            lost_frames=target.lost_frames,
        )


# ----------------------------------------------------------------------
# Fakes, for tests and replay. No hardware, no App Lab runtime.
# ----------------------------------------------------------------------
class FakeBridge:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.sensors = {"program_enabled": True, "ultrasonic_cm": 100}

    def call(self, method: str, *args):
        self.calls.append((method,) + args)
        if method == "read_sensors":
            import json

            return json.dumps(self.sensors)
        return True

    @property
    def last_drive(self) -> tuple | None:
        for entry in reversed(self.calls):
            if entry[0] == "drive_raw":
                return entry
        return None


class ScriptedVision:
    """Replays a list of WorldObservation-producing callables."""

    def __init__(self, frames: list) -> None:
        self._frames = frames
        self._index = 0

    def observe(self) -> WorldObservation:
        if self._index >= len(self._frames):
            return WorldObservation(
                timestamp_ms=monotonic_ms(), healthy=False, health_reason="end_of_script"
            )
        frame = self._frames[self._index]
        self._index += 1
        return frame


def observation(
    timestamp_ms: float,
    ball_x: float | None = None,
    ball_y: float = 0.3,
    ball_confidence: float = 0.9,
    goal_x: float | None = None,
    confirmed: bool = True,
    healthy: bool = True,
    age_ms: float = 0.0,
    ultrasonic_cm: int = -1,
    facing_side: str = "UNKNOWN",
    team_is_blue: bool = False,
) -> WorldObservation:
    """Terse WorldObservation builder for tests."""
    ball = TrackedTarget()
    if ball_x is not None:
        ball = TrackedTarget(
            visible=True,
            confirmed=confirmed,
            confidence=ball_confidence,
            x=ball_x,
            y=ball_y,
            age_ms=age_ms,
            detected_frames=3,
        )
    else:
        ball = TrackedTarget(visible=False, confirmed=False, age_ms=age_ms, lost_frames=3)

    goal = TrackedTarget()
    if goal_x is not None:
        goal = TrackedTarget(
            visible=True, confirmed=True, confidence=0.9, x=goal_x, y=0.3, age_ms=0.0
        )
    else:
        goal = TrackedTarget(visible=False, confirmed=False, age_ms=float("inf"))

    return WorldObservation(
        timestamp_ms=timestamp_ms,
        ball=ball,
        goal=goal,
        healthy=healthy,
        ultrasonic_cm=ultrasonic_cm,
        facing_side=facing_side,
        team_is_blue=team_is_blue,
    )
