"""Proportional steering and speed staging."""
from .config import Config
from .types import MotorCommand


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


class SteeringController:
    """Converts a horizontal error into a differential command.

    Sign convention: error > 0 means the target is right of centre, which
    must produce a right turn (left wheels faster).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._previous_error = 0.0

    def reset(self) -> None:
        self._previous_error = 0.0

    def turn_for(self, error: float, dt_s: float) -> float:
        cfg = self._config
        if abs(error) <= cfg.center_deadband:
            error = 0.0

        turn = cfg.kp_steer * error
        if cfg.kd_steer and dt_s > 0.0:
            turn += cfg.kd_steer * (error - self._previous_error) / dt_s
        self._previous_error = error
        return clamp(turn, -cfg.max_turn, cfg.max_turn)

    def command(self, base_speed: float, error: float, dt_s: float, reason: str) -> MotorCommand:
        cfg = self._config
        turn = self.turn_for(error, dt_s)
        limit = cfg.max_motor_command
        return MotorCommand(
            left=clamp(base_speed + turn, -limit, limit),
            right=clamp(base_speed - turn, -limit, limit),
            reason=reason,
        )

    def rotate(self, direction: int, speed: float, reason: str) -> MotorCommand:
        """Spin in place. direction +1 turns right, -1 turns left."""
        limit = self._config.max_motor_command
        speed = clamp(speed, 0.0, limit)
        return MotorCommand(left=direction * speed, right=-direction * speed, reason=reason)


def speed_from_ball_y(ball_y: float, config: Config) -> float:
    """Coarse distance proxy: a lower ball in frame means it is closer."""
    if ball_y < config.ball_y_far:
        return config.chase_speed_fast
    if ball_y < config.ball_y_close:
        return config.chase_speed_medium
    return config.chase_speed_close
