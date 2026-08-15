"""Validated tuning configuration. Every constant lives here, nowhere else.

Values marked MEASURED were established on this robot; the rest are the
specification's starting values and still need field tuning.
"""
import os
from dataclasses import dataclass, fields


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


class ConfigError(ValueError):
    """Raised when configuration is invalid. Prevents arming."""


@dataclass
class Config:
    # --- labels: MEASURED, these are this model's actual class names ---
    ball_label: str = _s("ROBOCUP_LABEL_BALL", "soccer_ball")
    goal_label: str = _s("ROBOCUP_LABEL_GOAL", "goal")
    opponent_label: str = _s("ROBOCUP_LABEL_ROBOT", "robot")

    # Goal positioning is OFF by default: this model has a single "goal" class
    # and cannot tell the attacking goal from our own. Enabling it without a
    # distinguishing rule risks scoring an own goal.
    enable_goal_detection: bool = _b("ROBOCUP_ENABLE_GOAL", False)
    enable_opponent_avoidance: bool = _b("ROBOCUP_ENABLE_OPPONENT", False)

    # --- detection gating ---
    # MEASURED: the runner's own threshold must also be lowered (see
    # runner_threshold); at the model's shipped 0.5 the ball only registers
    # within a few cm of the lens.
    runner_threshold: float = _f("ROBOCUP_RUNNER_THRESHOLD", 0.15)
    # Matched to runner_threshold: anything between the two was being computed
    # by the model and then thrown away, costing tracking continuity. The
    # tracker's confirm/loss streaks are the real false-positive defence, not
    # this threshold.
    ball_confidence_min: float = _f("ROBOCUP_BALL_CONF", 0.15)
    goal_confidence_min: float = _f("ROBOCUP_GOAL_CONF", 0.45)
    opponent_confidence_min: float = _f("ROBOCUP_OPP_CONF", 0.50)
    detection_confirm_frames: int = _i("ROBOCUP_CONFIRM_FRAMES", 2)
    align_stable_frames: int = _i("ROBOCUP_ALIGN_FRAMES", 2)
    attack_stable_frames: int = _i("ROBOCUP_ATTACK_FRAMES", 2)
    # MEASURED stationary: 5.75 FPS, gaps median 98ms / p90 227ms / MAX 1792ms.
    # The multi-second tail is present even when parked, so it is not motion -
    # it is RF congestion (16 competing team camera APs on 2.4GHz). At 1500ms
    # ~21% of frames tripped the health check and the robot spent its time
    # stopping instead of playing. 2400ms clears the measured tail.
    # Trade-off: the robot can drive blind for up to 2.4s. That is deliberate -
    # a genuinely dead camera never recovers, so the check still catches it,
    # while a congestion stall resolves on its own.
    vision_stale_ms: float = _f("ROBOCUP_STALE_MS", 2400.0)
    target_filter_alpha: float = _f("ROBOCUP_FILTER_ALPHA", 0.55)

    # --- geometry / tolerances ---
    align_tolerance: float = _f("ROBOCUP_ALIGN_TOL", 0.10)
    attack_ball_tolerance: float = _f("ROBOCUP_ATTACK_BALL_TOL", 0.10)
    attack_goal_tolerance: float = _f("ROBOCUP_ATTACK_GOAL_TOL", 0.18)
    ball_y_far: float = _f("ROBOCUP_BALL_Y_FAR", 0.45)
    ball_y_close: float = _f("ROBOCUP_BALL_Y_CLOSE", 0.72)
    get_behind_gain: float = _f("ROBOCUP_GET_BEHIND_GAIN", 0.60)
    get_behind_max_offset: float = _f("ROBOCUP_GET_BEHIND_MAX", 0.20)
    positioning_timeout_ms: float = _f("ROBOCUP_POSITIONING_TIMEOUT_MS", 1500.0)

    # --- steering ---
    kp_steer: float = _f("ROBOCUP_KP", 1.6)
    kd_steer: float = _f("ROBOCUP_KD", 0.0)
    max_turn: float = _f("ROBOCUP_MAX_TURN", 0.65)
    center_deadband: float = _f("ROBOCUP_DEADBAND", 0.025)
    max_motor_command: float = _f("ROBOCUP_MAX_MOTOR", 1.0)
    # MEASURED by ROBOCUP_MODE=MOTORS_RAMP at battery_mv 12015: the wheels
    # turn from the 0.20-0.30 band, so 0.25 is a safe floor. Commands below it
    # are scaled up (both sides together, preserving the turn ratio) so a slow
    # manoeuvre cannot silently produce no motion.
    # NOTE: this floor is battery-dependent. An earlier ramp at 4901mV (flat
    # pack) produced no motion at ANY level up to 0.70 - if slow moves stop
    # working, check battery_mv before re-tuning this.
    min_effective_speed: float = _f("ROBOCUP_MIN_SPEED", 0.25)
    motor_slew_per_second: float = _f("ROBOCUP_SLEW", 3.0)
    align_base_speed: float = _f("ROBOCUP_ALIGN_SPEED", 0.10)
    chase_speed_fast: float = _f("ROBOCUP_CHASE_FAST", 0.85)
    chase_speed_medium: float = _f("ROBOCUP_CHASE_MED", 0.65)
    chase_speed_close: float = _f("ROBOCUP_CHASE_CLOSE", 0.42)

    # --- search / recovery / attack ---
    default_search_direction: int = _i("ROBOCUP_SEARCH_DIR", 1)
    search_turn_speed: float = _f("ROBOCUP_SEARCH_SPEED", 0.30)
    search_sweep_ms: float = _f("ROBOCUP_SEARCH_SWEEP_MS", 1800.0)
    # MEASURED: continuous rotation is self-defeating on this robot. Spinning
    # starves the ESP32's shared antenna (frame gaps hit 1.5-1.7s vs 98ms
    # median stationary), and at 5.75 FPS the robot sweeps past the ball
    # between frames. Step-and-scan instead: a short turn, then hold still
    # long enough to capture a clean frame.
    search_pulse_ms: float = _f("ROBOCUP_SEARCH_PULSE_MS", 260.0)
    search_settle_ms: float = _f("ROBOCUP_SEARCH_SETTLE_MS", 450.0)
    # MEASURED: must exceed a single slow camera frame. The link degrades under
    # motion (see vision_stale_ms), so 700ms avoids treating ordinary driving
    # jitter as having lost the ball.
    lost_grace_ms: float = _f("ROBOCUP_LOST_GRACE_MS", 700.0)
    recover_turn_speed: float = _f("ROBOCUP_RECOVER_SPEED", 0.38)
    recover_timeout_ms: float = _f("ROBOCUP_RECOVER_TIMEOUT_MS", 2500.0)
    last_seen_memory_ms: float = _f("ROBOCUP_LAST_SEEN_MS", 5000.0)
    attack_speed: float = _f("ROBOCUP_ATTACK_SPEED", 1.0)
    attack_burst_ms: float = _f("ROBOCUP_ATTACK_BURST_MS", 500.0)
    attack_cooldown_ms: float = _f("ROBOCUP_ATTACK_COOLDOWN_MS", 700.0)

    # --- loop / safety ---
    # MEASURED: no value in running much above the 5.75 FPS camera. Repeated
    # frames are cheap (inference is skipped), so 10Hz keeps steering
    # responsive without burning CPU.
    control_loop_target_hz: float = _f("ROBOCUP_LOOP_HZ", 10.0)
    command_watchdog_ms: int = _i("ROBOCUP_WATCHDOG_MS", 400)
    telemetry_hz: float = _f("ROBOCUP_TELEMETRY_HZ", 5.0)
    arm_on_start: bool = _b("ROBOCUP_ARM_ON_START", False)

    # Ultrasonic is an extra contact cue this robot has that the spec did not
    # assume. 0 disables it; the ball is only "at the bumper" below this.
    contact_cm: int = _i("ROBOCUP_CONTACT_CM", 0)

    # --- field-side awareness (fixes own goals) ---
    # The two goals look identical, so the model cannot tell them apart. The
    # red/blue wall tape is the only signal that differs between the ends.
    enable_field_side: bool = _b("ROBOCUP_ENABLE_FIELD_SIDE", True)

    # CHECK THIS FIRST AT THE VENUE. If true, a RED team defends the RED-taped
    # end. If the tournament marks the end you ATTACK instead, set
    # ROBOCUP_OWN_WALL_IS_TEAM=false - getting it backwards doubles own goals
    # rather than preventing them. Verify with ROBOCUP_MODE=FIELD_SIDE_CHECK.
    own_wall_is_team_colour: bool = _b("ROBOCUP_OWN_WALL_IS_TEAM", True)

    # Team override for when the CAM button toggle is unreliable:
    # "auto" reads hold_toggle, or force "red" / "blue".
    team_override: str = _s("ROBOCUP_TEAM", "auto")

    # Only the upper part of the frame is wall; the rest is floor and ball.
    wall_band_fraction: float = _f("ROBOCUP_WALL_BAND", 0.45)
    wall_saturation_min: int = _i("ROBOCUP_WALL_SAT_MIN", 120)
    wall_value_min: int = _i("ROBOCUP_WALL_VAL_MIN", 70)
    wall_min_coverage: float = _f("ROBOCUP_WALL_MIN_COVERAGE", 0.02)
    # One colour must beat the other by this factor to count as a decision.
    wall_dominance_ratio: float = _f("ROBOCUP_WALL_DOMINANCE", 1.8)
    # A single mislabelled frame must never flip our shooting direction.
    wall_confirm_frames: int = _i("ROBOCUP_WALL_CONFIRM_FRAMES", 3)

    # When the side cannot be determined, is shooting allowed? False is the
    # safe default: no shot is better than a 50/50 own goal. Set true only if
    # the robot turns out to be too passive at the venue.
    attack_when_side_unknown: bool = _b("ROBOCUP_ATTACK_IF_UNKNOWN", False)

    # Repositioning: back off the ball, then turn to face the other end.
    reposition_backoff_ms: float = _f("ROBOCUP_REPOSITION_BACKOFF_MS", 450.0)
    reposition_speed: float = _f("ROBOCUP_REPOSITION_SPEED", 0.45)
    reposition_timeout_ms: float = _f("ROBOCUP_REPOSITION_TIMEOUT_MS", 4000.0)
    # How sharply the arc curves: 0 = spin in place, 1 = straight line.
    reposition_arc_ratio: float = _f("ROBOCUP_REPOSITION_ARC", 0.25)
    # After a failed reposition, do not immediately retry it.
    reposition_cooldown_ms: float = _f("ROBOCUP_REPOSITION_COOLDOWN_MS", 3000.0)

    def validate(self) -> None:
        """Raise ConfigError on any invalid value. Called before arming."""
        errors: list[str] = []

        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, float) and value != value:
                errors.append(f"{f.name} is NaN")

        def check(cond: bool, message: str) -> None:
            if not cond:
                errors.append(message)

        check(0.0 < self.runner_threshold <= 1.0, "runner_threshold must be in (0, 1]")
        for name in ("ball_confidence_min", "goal_confidence_min", "opponent_confidence_min"):
            check(0.0 <= getattr(self, name) <= 1.0, f"{name} must be in [0, 1]")
        check(
            self.ball_confidence_min >= self.runner_threshold,
            "ball_confidence_min below runner_threshold has no effect: the runner "
            "filters boxes before returning them",
        )
        check(self.detection_confirm_frames >= 1, "detection_confirm_frames must be >= 1")
        check(self.align_stable_frames >= 1, "align_stable_frames must be >= 1")
        check(self.attack_stable_frames >= 1, "attack_stable_frames must be >= 1")
        check(0.0 < self.target_filter_alpha <= 1.0, "target_filter_alpha must be in (0, 1]")
        check(0.0 < self.align_tolerance < 0.5, "align_tolerance must be in (0, 0.5)")
        check(
            0.0 < self.ball_y_far < self.ball_y_close < 1.0,
            "require 0 < ball_y_far < ball_y_close < 1",
        )
        check(self.kp_steer > 0.0, "kp_steer must be > 0")
        check(0.0 < self.max_turn <= 1.0, "max_turn must be in (0, 1]")
        check(0.0 <= self.center_deadband < self.align_tolerance,
              "center_deadband must be >= 0 and below align_tolerance")
        check(0.0 < self.max_motor_command <= 1.0, "max_motor_command must be in (0, 1]")
        check(self.motor_slew_per_second > 0.0, "motor_slew_per_second must be > 0")
        for name in ("align_base_speed", "chase_speed_fast", "chase_speed_medium",
                     "chase_speed_close", "search_turn_speed", "recover_turn_speed",
                     "attack_speed"):
            check(0.0 <= getattr(self, name) <= 1.0, f"{name} must be in [0, 1]")
        check(self.default_search_direction in (-1, 1), "default_search_direction must be -1 or 1")
        check(self.attack_burst_ms > 0.0, "attack_burst_ms must be > 0")
        check(self.vision_stale_ms > 0.0, "vision_stale_ms must be > 0")
        check(self.control_loop_target_hz > 0.0, "control_loop_target_hz must be > 0")
        check(
            self.command_watchdog_ms > (1000.0 / self.control_loop_target_hz),
            "command_watchdog_ms must exceed one control period or the motors "
            "will stutter between commands",
        )
        check(self.command_watchdog_ms <= 5000, "command_watchdog_ms exceeds the firmware cap of 5000")

        if errors:
            raise ConfigError("invalid configuration: " + "; ".join(errors))

    def describe(self) -> str:
        return (
            f"ball='{self.ball_label}' goal='{self.goal_label}' "
            f"goal_positioning={'ON' if self.enable_goal_detection else 'OFF'} "
            f"opponent={'ON' if self.enable_opponent_avoidance else 'OFF'} "
            f"conf>={self.ball_confidence_min} kp={self.kp_steer} "
            f"loop={self.control_loop_target_hz}Hz"
        )
