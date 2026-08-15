"""Data contracts shared by every stage of the soccer controller."""
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    SEARCH = "SEARCH"
    ALIGN = "ALIGN"
    CHASE = "CHASE"
    ATTACK = "ATTACK"
    RECOVER = "RECOVER"
    # Entered when the ball is close but we are aimed at our OWN goal. Backs
    # off and turns to the other end rather than scoring against ourselves.
    REPOSITION = "REPOSITION"


@dataclass
class Detection:
    """One normalized detection. x/y are 0..1, origin top-left."""

    label: str
    confidence: float
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0

    def is_valid(self) -> bool:
        """Reject NaN, infinities and out-of-frame coordinates."""
        for value in (self.confidence, self.x, self.y):
            if value != value or value in (float("inf"), float("-inf")):
                return False
        return 0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0


@dataclass
class TrackedTarget:
    """Smoothed, streak-gated view of one label over time."""

    visible: bool = False
    confirmed: bool = False
    confidence: float = 0.0
    x: float = 0.5
    y: float = 0.5
    age_ms: float = float("inf")
    detected_frames: int = 0
    lost_frames: int = 0

    def is_fresh(self, stale_ms: float) -> bool:
        return self.confirmed and self.age_ms <= stale_ms


@dataclass
class WorldObservation:
    timestamp_ms: float = 0.0
    frame_id: int = 0
    inference_ms: float = 0.0
    healthy: bool = True
    health_reason: str = ""
    ball: TrackedTarget = field(default_factory=TrackedTarget)
    goal: TrackedTarget = field(default_factory=TrackedTarget)
    opponent: TrackedTarget = field(default_factory=TrackedTarget)
    ultrasonic_cm: int = -1
    # Wall colour ahead: "RED" / "BLUE" / "UNKNOWN". This is what tells us
    # which end a forward push would send the ball toward.
    facing_side: str = "UNKNOWN"
    team_is_blue: bool = False


@dataclass
class MotorCommand:
    """Normalized differential command. The adapter maps it onto four wheels."""

    left: float = 0.0
    right: float = 0.0
    reason: str = "init"

    @staticmethod
    def stop(reason: str) -> "MotorCommand":
        return MotorCommand(0.0, 0.0, reason)

    def is_stop(self) -> bool:
        return self.left == 0.0 and self.right == 0.0
