"""Autonomous soccer entry point.

Modes (ROBOCUP_MODE):
  VISION_ONLY      inference + telemetry, motors forced to zero (default)
  FIELD_SIDE_CHECK confirm which wall colour means which goal (no motion)
  MOTORS_ON_BLOCKS low-power polarity check, wheels raised
  MOTORS_RAMP      find the minimum speed that turns the wheels
  FIELD_RUN        live control

Arming is the CAM BOOT button: the firmware's program_enabled flag. The robot
never drives while disarmed, and pressing the button again stops it.
"""
import json
import os
import time

from arduino.app_utils import App, Bridge

from soccer.adapters import MotorAdapter, VisionAdapter, monotonic_ms
from soccer.config import Config, ConfigError
from soccer.controller import SoccerController
from soccer.safety import SafetyLimiter
from soccer.telemetry import Telemetry
from soccer.field import Side, WallDetector, attack_side, own_side
from soccer.tracker import TargetTracker
from soccer.types import MotorCommand, State
from vision import CameraStream, Detector

MODE = os.environ.get("ROBOCUP_MODE", "VISION_ONLY").upper()

config = Config()
try:
    config.validate()
except ConfigError as exc:
    print(f"[FATAL] {exc}")
    raise SystemExit(1)

print("=" * 62)
print(f"AUTONOMOUS SOCCER - mode={MODE}")
print(f"config: {config.describe()}")
print("=" * 62)
if not config.enable_goal_detection:
    print(
        "[NOTE] goal positioning is OFF: this model has one 'goal' class and "
        "cannot tell our net from theirs. Ball-only pursuit is used."
    )

telemetry = Telemetry(config.telemetry_hz)
limiter = SafetyLimiter(config)
motors = MotorAdapter(config, Bridge)
controller = SoccerController(config)

camera = CameraStream()
camera.start()

wall = WallDetector(config) if config.enable_field_side else None

trackers = {
    "ball": TargetTracker(config.ball_label, config.ball_confidence_min, config),
    "goal": TargetTracker(config.goal_label, config.goal_confidence_min, config),
    "opponent": TargetTracker(
        config.opponent_label, config.opponent_confidence_min, config
    ),
}

_detector_cm = Detector()
detector = _detector_cm.__enter__()
detector.set_threshold(config.runner_threshold)
vision = VisionAdapter(config, camera, detector, trackers, wall)

_last_step = time.monotonic()
_last_transition_count = 0
_armed = False
_period = 1.0 / config.control_loop_target_hz


def _resolve_team() -> bool:
    """True = BLUE. Honours ROBOCUP_TEAM, else the CAM button hold toggle."""
    override = config.team_override.strip().lower()
    if override in ("red", "blue"):
        return override == "blue"
    try:
        raw = Bridge.call("read_sensors")
        return bool(json.loads(raw).get("hold_toggle")) if raw else False
    except Exception:  # noqa: BLE001
        return False


def field_side_check() -> None:
    """Point the robot at each end and confirm the mapping before playing.

    Getting own_wall_is_team_colour backwards doubles own goals rather than
    preventing them, so this must be verified, never assumed.
    """
    team_blue = _resolve_team()
    team = "BLUE" if team_blue else "RED"
    print("=" * 62)
    print(f"FIELD SIDE CHECK - team {team}  (no motion)")
    print(f"  defending: {own_side(team_blue, config).value} wall")
    print(f"  attacking: {attack_side(team_blue, config).value} wall")
    print("  Point the robot at ONE end, then the other. 40 s.")
    print("=" * 62)
    end = time.monotonic() + 40
    last = ""
    while time.monotonic() < end:
        frame, age = camera.latest()
        if frame is None or age > 2.0:
            time.sleep(0.2)
            continue
        reading = wall.update(frame)
        verdict = "OWN GOAL - would not shoot" if wall.side is own_side(team_blue, config) \
            else ("THEIR GOAL - would shoot" if wall.side is not Side.UNKNOWN else "unsure")
        line = f"{reading.describe():<44} stable={wall.side.value:<8} {verdict}"
        if line != last:
            last = line
            print(f"  {line}")
        time.sleep(0.25)
    print("FIELD SIDE CHECK complete")


def _is_armed() -> bool:
    """Arming is the physical BOOT button, read through the firmware."""
    try:
        raw = Bridge.call("read_sensors")
        return bool(json.loads(raw).get("program_enabled")) if raw else False
    except Exception:  # noqa: BLE001 - a failed read must never mean 'armed'
        return False


def motors_on_blocks() -> None:
    """Manual low-power polarity check. Wheels must be raised."""
    print("[BLOCKS] wheels must be raised. 6 moves at low power.")
    for left, right, label in (
        (0.25, 0.25, "forward"),
        (-0.25, -0.25, "backward"),
        (0.25, -0.25, "rotate right"),
        (-0.25, 0.25, "rotate left"),
    ):
        print(f"[BLOCKS] {label}")
        motors.apply(MotorCommand(left, right, label))
        time.sleep(1.2)
        motors.stop("between")
        time.sleep(0.8)
    print("[BLOCKS] done")


RAMP_LEVELS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]


def motors_ramp() -> None:
    """Find the minimum speed that actually turns the wheels.

    Steps forward at increasing power with a clear pause between each, so the
    operator can count which step first moved. Bypasses the stiction floor on
    purpose - measuring it is the point.
    """
    print("[RAMP] wheels raised. 9 forward steps, ~1.2s each, 1s pause between.")
    for index, level in enumerate(RAMP_LEVELS, start=1):
        print(f"[RAMP] step {index}/9  power={level:.2f}")
        raw = int(round(level * 255))
        Bridge.call("drive_raw", raw, raw, raw, raw, 1200)
        time.sleep(1.2)
        Bridge.call("drive_raw", 0, 0, 0, 0, 0)
        time.sleep(1.0)
    print("[RAMP] done. Report the FIRST step number where the wheels turned.")


def loop() -> None:
    global _last_step, _armed, _last_transition_count

    now = time.monotonic()
    dt_s = max(now - _last_step, 1e-3)
    _last_step = now

    try:
        vision.team_is_blue = _resolve_team()
        obs = vision.observe()

        armed_now = MODE == "FIELD_RUN" and _is_armed()
        if armed_now != _armed:
            _armed = armed_now
            telemetry.event(f"{'ARMED' if _armed else 'DISARMED'}")
            if not _armed:
                motors.stop("disarmed")
                limiter.emergency_stop("disarmed")

        command = controller.step(obs, dt_s)

        # Report transitions the controller recorded this iteration.
        while _last_transition_count < len(controller.transitions):
            entry = controller.transitions[_last_transition_count]
            telemetry.transition(*entry)
            _last_transition_count += 1

        if not _armed:
            command = MotorCommand.stop("disarmed" if MODE == "FIELD_RUN" else MODE.lower())
            motors.stop(command.reason)
        else:
            command = limiter.limit(command, dt_s)
            motors.apply(command)

        telemetry.emit(controller.state.value, obs, command, _armed)

    except Exception as exc:  # noqa: BLE001 - never let the loop die driving
        motors.stop("controller_exception")
        limiter.emergency_stop("controller_exception")
        _armed = False
        telemetry.error(f"controller_exception: {type(exc).__name__}: {exc}")

    # Hold the target loop rate without busy-waiting.
    remaining = _period - (time.monotonic() - now)
    if remaining > 0:
        time.sleep(remaining)


# A polarity check is about wheels, not vision, so it must not depend on the
# camera being up.
if MODE == "FIELD_SIDE_CHECK":
    print("[INFO] waiting for camera...")
    _d = time.monotonic() + 20
    while not camera.is_fresh() and time.monotonic() < _d:
        time.sleep(0.25)
    try:
        if not camera.is_fresh():
            print("[FATAL] no camera frames.")
        else:
            field_side_check()
    finally:
        motors.stop("shutdown")
        camera.stop()
        _detector_cm.__exit__(None, None, None)
    raise SystemExit(0)

if MODE in ("MOTORS_ON_BLOCKS", "MOTORS_RAMP"):
    try:
        motors_ramp() if MODE == "MOTORS_RAMP" else motors_on_blocks()
    finally:
        motors.stop("shutdown")
        camera.stop()
        _detector_cm.__exit__(None, None, None)
    raise SystemExit(0)

print("[INFO] waiting for camera...")
_deadline = time.monotonic() + 20
while not camera.is_fresh() and time.monotonic() < _deadline:
    time.sleep(0.25)
if not camera.is_fresh():
    print("[FATAL] no camera frames. Is wlan0 joined to the camera AP?")
    camera.stop()
    raise SystemExit(1)
print("[INFO] camera ready")

if MODE == "FIELD_RUN":
    print("[INFO] press the CAM BOOT button to ARM. Press again to stop.")
else:
    print("[INFO] VISION_ONLY: motors are forced to zero.")

try:
    motors.stop("startup")
    App.run(user_loop=loop)
finally:
    motors.stop("shutdown")
    camera.stop()
    _detector_cm.__exit__(None, None, None)
    print(f"[INFO] stopped. loop rate {telemetry.rate_hz():.1f} Hz")
