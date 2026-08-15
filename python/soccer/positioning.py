"""Goal-relative positioning: approach the ball from the side that lines up
`robot -> ball -> goal` before pushing.

DANGER: this is only correct if `goal` means the ATTACKING goal. This robot's
model has a single "goal" class and cannot distinguish the two, so
Config.enable_goal_detection defaults to False. Turning it on without a
distinguishing rule can aim the ball at our own net.
"""
from .config import Config
from .steering import clamp
from .types import TrackedTarget


def desired_ball_x(ball: TrackedTarget, goal: TrackedTarget, config: Config) -> float:
    """Where the ball should sit in frame so that pushing it sends it goalward.

    To send the ball at the goal we must be opposite the goal across the ball
    (robot -> ball -> goal). Goal right of the ball means we approach the
    ball's LEFT side, so the ball should sit RIGHT of centre in our view.

    NOTE: this deliberately uses `0.5 + offset` where the written spec says
    `0.5 - offset`. The spec's own prose ("goal right of the ball ... approach
    the ball's left side") contradicts its formula: with a minus sign the
    robot steers to the ball's right and pushes it away from the goal. Sign
    verified by test_positioning.
    """
    goal_relative_x = goal.x - ball.x
    offset = clamp(
        config.get_behind_gain * goal_relative_x,
        -config.get_behind_max_offset,
        config.get_behind_max_offset,
    )
    return 0.5 + offset


def positioning_error(ball: TrackedTarget, goal: TrackedTarget, config: Config) -> float:
    return ball.x - desired_ball_x(ball, goal, config)


def should_position(ball: TrackedTarget, goal: TrackedTarget, config: Config) -> bool:
    """Only worth positioning when the ball is close and the goal is fresh."""
    if not config.enable_goal_detection:
        return False
    if not goal.is_fresh(config.vision_stale_ms):
        return False
    return ball.y >= config.ball_y_far
