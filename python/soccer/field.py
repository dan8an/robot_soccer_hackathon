"""Which end of the field are we facing?

The two goals are physically identical, so no vision model can tell them
apart - a classifier cannot learn a difference that does not exist in the
pixels. The only thing that differs between the ends is the coloured tape on
the walls: red on one side, blue on the other.

So we classify the wall behind the ball by hue. That gives the one bit the
controller was missing: is a forward push aimed at their net or ours.
"""
from dataclasses import dataclass
from enum import Enum

from .config import Config


class Side(str, Enum):
    RED = "RED"
    BLUE = "BLUE"
    UNKNOWN = "UNKNOWN"


@dataclass
class WallReading:
    side: Side
    red_fraction: float
    blue_fraction: float

    def describe(self) -> str:
        return (
            f"{self.side.value} (red={self.red_fraction*100:.1f}% "
            f"blue={self.blue_fraction*100:.1f}%)"
        )


class WallDetector:
    """Classifies the wall colour in the upper part of the frame.

    Only the upper band is sampled: the lower frame is floor and the ball, and
    including them drags the ratios toward whatever the carpet happens to be.

    A hysteresis counter guards the output. A single mislabelled frame must not
    flip our idea of which way we are shooting - that is exactly the error that
    scores an own goal.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._candidate = Side.UNKNOWN
        self._streak = 0
        self.side = Side.UNKNOWN
        self.last: WallReading | None = None

    def update(self, frame) -> WallReading:
        reading = self._classify(frame)
        self.last = reading

        if reading.side is self._candidate:
            self._streak += 1
        else:
            self._candidate = reading.side
            self._streak = 1

        # UNKNOWN is adopted immediately (losing sight of the wall is real
        # information), but a colour must survive the streak to take effect.
        if reading.side is Side.UNKNOWN:
            if self._streak >= self._config.wall_confirm_frames:
                self.side = Side.UNKNOWN
        elif self._streak >= self._config.wall_confirm_frames:
            self.side = reading.side

        return reading

    def _classify(self, frame) -> WallReading:
        import cv2
        import numpy as np

        cfg = self._config
        rgb = np.asarray(frame, dtype=np.uint8)
        height = rgb.shape[0]
        band = rgb[: max(1, int(height * cfg.wall_band_fraction)), :, :]

        hsv = cv2.cvtColor(band[:, :, ::-1], cv2.COLOR_BGR2HSV)
        sat_min = cfg.wall_saturation_min
        val_min = cfg.wall_value_min

        # Red wraps the hue origin, so it needs two ranges combined.
        red_low = cv2.inRange(hsv, (0, sat_min, val_min), (10, 255, 255))
        red_high = cv2.inRange(hsv, (160, sat_min, val_min), (179, 255, 255))
        red_mask = cv2.bitwise_or(red_low, red_high)
        blue_mask = cv2.inRange(hsv, (100, sat_min, val_min), (130, 255, 255))

        total = float(band.shape[0] * band.shape[1]) or 1.0
        red_fraction = float(cv2.countNonZero(red_mask)) / total
        blue_fraction = float(cv2.countNonZero(blue_mask)) / total

        floor = cfg.wall_min_coverage
        side = Side.UNKNOWN
        if red_fraction >= floor or blue_fraction >= floor:
            # Require a clear winner; a near-tie means we are probably looking
            # at a corner or at neither wall.
            stronger, weaker = max(red_fraction, blue_fraction), min(red_fraction, blue_fraction)
            if stronger >= weaker * cfg.wall_dominance_ratio:
                side = Side.RED if red_fraction > blue_fraction else Side.BLUE

        return WallReading(side=side, red_fraction=red_fraction, blue_fraction=blue_fraction)


def own_side(team_is_blue: bool, config: Config) -> Side:
    """Which wall colour marks the end we are DEFENDING."""
    team = Side.BLUE if team_is_blue else Side.RED
    if config.own_wall_is_team_colour:
        return team
    return Side.RED if team is Side.BLUE else Side.BLUE


def attack_side(team_is_blue: bool, config: Config) -> Side:
    """Which wall colour marks the end we are SHOOTING AT."""
    ours = own_side(team_is_blue, config)
    return Side.RED if ours is Side.BLUE else Side.BLUE


def facing_own_goal(facing: Side, team_is_blue: bool, config: Config) -> bool:
    """True only when we are certain we are aimed at our own net.

    UNKNOWN deliberately returns False: see Config.attack_when_side_unknown for
    how an unknown side is handled, which is a separate decision from this one.
    """
    if facing is Side.UNKNOWN:
        return False
    return facing is own_side(team_is_blue, config)
