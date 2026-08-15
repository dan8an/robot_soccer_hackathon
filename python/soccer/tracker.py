"""Turns per-frame detections into streak-gated, filtered targets."""
from .config import Config
from .types import Detection, TrackedTarget


class TargetTracker:
    """Tracks one label across frames.

    Confirmation requires consecutive valid detections, so a single spurious
    box cannot trigger a state change. Position is exponentially filtered;
    the raw confidence and age are not, because safety decisions must react
    immediately.
    """

    def __init__(self, label: str, confidence_min: float, config: Config) -> None:
        self.label = label
        self.confidence_min = confidence_min
        self._config = config
        self._filtered_x: float | None = None
        self._filtered_y: float | None = None
        self._detected_frames = 0
        self._lost_frames = 0
        self._last_seen_ms: float | None = None
        self._last_confidence = 0.0

    def update(self, detections: list[Detection], now_ms: float) -> TrackedTarget:
        best = self._select(detections)

        if best is None:
            self._lost_frames += 1
            self._detected_frames = 0
            age = (
                float("inf")
                if self._last_seen_ms is None
                else max(0.0, now_ms - self._last_seen_ms)
            )
            # Memory expires: an old position must never look like a detection.
            if age > self._config.last_seen_memory_ms:
                self._filtered_x = None
                self._filtered_y = None
            return TrackedTarget(
                visible=False,
                confirmed=False,
                confidence=0.0,
                x=self._filtered_x if self._filtered_x is not None else 0.5,
                y=self._filtered_y if self._filtered_y is not None else 0.5,
                age_ms=age,
                detected_frames=0,
                lost_frames=self._lost_frames,
            )

        alpha = self._config.target_filter_alpha
        if self._filtered_x is None:
            self._filtered_x, self._filtered_y = best.x, best.y
        else:
            self._filtered_x = alpha * best.x + (1.0 - alpha) * self._filtered_x
            self._filtered_y = alpha * best.y + (1.0 - alpha) * self._filtered_y

        self._detected_frames += 1
        self._lost_frames = 0
        self._last_seen_ms = now_ms
        self._last_confidence = best.confidence

        return TrackedTarget(
            visible=True,
            confirmed=self._detected_frames >= self._config.detection_confirm_frames,
            confidence=best.confidence,
            x=self._filtered_x,
            y=self._filtered_y,
            age_ms=0.0,
            detected_frames=self._detected_frames,
            lost_frames=0,
        )

    def _select(self, detections: list[Detection]) -> Detection | None:
        """Highest-confidence valid candidate, biased toward the previous position.

        Unrelated detections are never averaged; one candidate wins outright.
        """
        candidates = [
            d
            for d in detections
            if d.label == self.label and d.confidence >= self.confidence_min and d.is_valid()
        ]
        if not candidates:
            return None
        if len(candidates) == 1 or self._filtered_x is None:
            return max(candidates, key=lambda d: d.confidence)

        def score(d: Detection) -> float:
            # Prefer confident detections near where the target was last seen,
            # so a distant false positive cannot steal the lock.
            distance = abs(d.x - self._filtered_x) + abs(d.y - (self._filtered_y or 0.5))
            return d.confidence - 0.35 * distance

        return max(candidates, key=score)


class BallMemory:
    """Directional memory used to recover quickly after losing the ball."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self.last_x: float | None = None
        self.last_y: float | None = None
        self.last_seen_ms: float | None = None
        self.last_side: int = config.default_search_direction

    def update(self, ball: TrackedTarget, now_ms: float) -> None:
        if not ball.visible:
            return
        self.last_x = ball.x
        self.last_y = ball.y
        self.last_seen_ms = now_ms
        error = ball.x - 0.5
        # Only update side when the ball is meaningfully off-centre; a centred
        # ball carries no directional information worth remembering.
        if abs(error) > self._config.center_deadband:
            self.last_side = 1 if error > 0 else -1

    def side(self, now_ms: float) -> int:
        """Last-seen side while memory is fresh, else the configured default."""
        if self.last_seen_ms is None:
            return self._config.default_search_direction
        if (now_ms - self.last_seen_ms) > self._config.last_seen_memory_ms:
            return self._config.default_search_direction
        return self.last_side
