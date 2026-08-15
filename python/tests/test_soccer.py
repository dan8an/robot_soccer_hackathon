"""Off-robot tests for the soccer controller. Run: python3 python/tests/test_soccer.py

Deliberately dependency-free (no pytest) so it runs anywhere, including on the
board itself.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer.adapters import FakeBridge, MotorAdapter, observation  # noqa: E402
from soccer.config import Config, ConfigError  # noqa: E402
from soccer.controller import SoccerController  # noqa: E402
from soccer.positioning import desired_ball_x  # noqa: E402
from soccer.safety import SafetyLimiter  # noqa: E402
from soccer.steering import SteeringController, speed_from_ball_y  # noqa: E402
from soccer.tracker import BallMemory, TargetTracker  # noqa: E402
from soccer.types import Detection, MotorCommand, State, TrackedTarget  # noqa: E402

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
    else:
        FAILED.append(f"{name}: {detail}")


def cfg(**overrides) -> Config:
    c = Config()
    for key, value in overrides.items():
        setattr(c, key, value)
    return c


# ----------------------------------------------------------------- config
def test_config() -> None:
    c = Config()
    try:
        c.validate()
        check("config/defaults valid", True)
    except ConfigError as exc:
        check("config/defaults valid", False, str(exc))

    bad = cfg(kp_steer=-1.0)
    try:
        bad.validate()
        check("config/rejects negative kp", False, "no error raised")
    except ConfigError:
        check("config/rejects negative kp", True)

    # A client filter below the runner threshold silently does nothing.
    bad = cfg(ball_confidence_min=0.1, runner_threshold=0.3)
    try:
        bad.validate()
        check("config/rejects conf below runner threshold", False, "no error raised")
    except ConfigError:
        check("config/rejects conf below runner threshold", True)

    # Watchdog shorter than a control period makes the motors stutter.
    bad = cfg(command_watchdog_ms=10, control_loop_target_hz=12.0)
    try:
        bad.validate()
        check("config/rejects short watchdog", False, "no error raised")
    except ConfigError:
        check("config/rejects short watchdog", True)


# --------------------------------------------------------------- steering
def test_steering() -> None:
    c = cfg()
    s = SteeringController(c)

    # Ball right of centre must turn right: left wheels faster than right.
    command = s.command(0.5, +0.3, 0.1, "t")
    check("steer/ball right turns right", command.left > command.right,
          f"L={command.left} R={command.right}")

    s.reset()
    command = s.command(0.5, -0.3, 0.1, "t")
    check("steer/ball left turns left", command.right > command.left,
          f"L={command.left} R={command.right}")

    s.reset()
    command = s.command(0.5, 0.01, 0.1, "t")
    check("steer/deadband ignores tiny error", command.left == command.right,
          f"L={command.left} R={command.right}")

    s.reset()
    turn = s.turn_for(10.0, 0.1)
    check("steer/turn clamped", abs(turn) <= c.max_turn, f"turn={turn}")

    s.reset()
    command = s.command(1.0, 1.0, 0.1, "t")
    check("steer/output clamped", max(abs(command.left), abs(command.right)) <= c.max_motor_command)

    # rotate(+1) must spin right: left forward, right backward.
    command = s.rotate(1, 0.3, "r")
    check("steer/rotate right sign", command.left > 0 > command.right)

    check("steer/speed far", speed_from_ball_y(0.1, c) == c.chase_speed_fast)
    check("steer/speed medium", speed_from_ball_y(0.6, c) == c.chase_speed_medium)
    check("steer/speed close", speed_from_ball_y(0.9, c) == c.chase_speed_close)


# ----------------------------------------------------------------- safety
def test_safety() -> None:
    c = cfg(motor_slew_per_second=1.0)
    limiter = SafetyLimiter(c)

    out = limiter.limit(MotorCommand(1.0, 1.0, "go"), 0.1)
    check("safety/slew limits ramp", out.left <= 0.11, f"left={out.left}")

    out = limiter.limit(MotorCommand(float("nan"), 0.5, "bad"), 0.1)
    check("safety/nan becomes stop", out.is_stop() and out.reason == "non_finite_command")

    out = limiter.limit(MotorCommand(float("inf"), 0.5, "bad"), 0.1)
    check("safety/inf becomes stop", out.is_stop())

    # A stop must be immediate, never ramped down through the slew limiter.
    limiter = SafetyLimiter(c)
    limiter.limit(MotorCommand(0.5, 0.5, "go"), 1.0)
    out = limiter.limit(MotorCommand.stop("estop"), 0.001)
    check("safety/stop bypasses slew", out.left == 0.0 and out.right == 0.0)

    out = limiter.limit(MotorCommand(5.0, -5.0, "over"), 10.0)
    check("safety/clamps to max", abs(out.left) <= c.max_motor_command)


# ---------------------------------------------------------------- tracker
def test_tracker() -> None:
    c = cfg(detection_confirm_frames=2, target_filter_alpha=1.0)
    t = TargetTracker("soccer_ball", 0.4, c)

    ball = Detection("soccer_ball", 0.9, 0.6, 0.5)
    out = t.update([ball], 0.0)
    check("tracker/not confirmed on first frame", out.visible and not out.confirmed)
    out = t.update([ball], 100.0)
    check("tracker/confirmed after streak", out.confirmed)

    out = t.update([], 200.0)
    check("tracker/loss clears visible", not out.visible and out.lost_frames == 1)
    check("tracker/age advances", out.age_ms == 100.0, f"age={out.age_ms}")

    # Low-confidence detections must not confirm anything.
    t2 = TargetTracker("soccer_ball", 0.4, c)
    out = t2.update([Detection("soccer_ball", 0.2, 0.5, 0.5)], 0.0)
    check("tracker/rejects low confidence", not out.visible)

    # NaN coordinates must be rejected outright.
    t3 = TargetTracker("soccer_ball", 0.4, c)
    out = t3.update([Detection("soccer_ball", 0.9, float("nan"), 0.5)], 0.0)
    check("tracker/rejects nan coords", not out.visible)

    # Out-of-frame coordinates are invalid.
    t4 = TargetTracker("soccer_ball", 0.4, c)
    out = t4.update([Detection("soccer_ball", 0.9, 1.4, 0.5)], 0.0)
    check("tracker/rejects out of frame", not out.visible)

    # Wrong label never matches.
    t5 = TargetTracker("soccer_ball", 0.4, c)
    out = t5.update([Detection("goal", 0.9, 0.5, 0.5)], 0.0)
    check("tracker/ignores other labels", not out.visible)

    # Filtering must smooth, not jump.
    c2 = cfg(target_filter_alpha=0.5, detection_confirm_frames=1)
    t6 = TargetTracker("soccer_ball", 0.4, c2)
    t6.update([Detection("soccer_ball", 0.9, 0.0, 0.5)], 0.0)
    out = t6.update([Detection("soccer_ball", 0.9, 1.0, 0.5)], 10.0)
    check("tracker/filters position", abs(out.x - 0.5) < 1e-6, f"x={out.x}")

    # The nearer candidate should win even at slightly lower confidence.
    c3 = cfg(target_filter_alpha=1.0, detection_confirm_frames=1)
    t7 = TargetTracker("soccer_ball", 0.4, c3)
    t7.update([Detection("soccer_ball", 0.9, 0.2, 0.5)], 0.0)
    out = t7.update(
        [Detection("soccer_ball", 0.85, 0.25, 0.5), Detection("soccer_ball", 0.9, 0.95, 0.5)],
        10.0,
    )
    check("tracker/prefers nearby candidate", abs(out.x - 0.25) < 1e-6, f"x={out.x}")


def test_memory() -> None:
    c = cfg()
    m = BallMemory(c)
    m.update(TrackedTarget(visible=True, x=0.8, y=0.5), 0.0)
    check("memory/records right side", m.side(100.0) == 1)
    m.update(TrackedTarget(visible=True, x=0.2, y=0.5), 200.0)
    check("memory/records left side", m.side(300.0) == -1)
    # Expired memory falls back to the configured default direction.
    check("memory/expires", m.side(200.0 + c.last_seen_memory_ms + 1) == c.default_search_direction)


# ----------------------------------------------------------- positioning
def test_positioning() -> None:
    c = cfg(enable_goal_detection=True)
    ball = TrackedTarget(visible=True, confirmed=True, x=0.5, y=0.8, age_ms=0.0)

    # Goal right of ball -> approach the ball's left -> want ball right of centre.
    goal_right = TrackedTarget(visible=True, confirmed=True, x=0.9, y=0.3, age_ms=0.0)
    check("position/goal right pushes target right", desired_ball_x(ball, goal_right, c) > 0.5)

    goal_left = TrackedTarget(visible=True, confirmed=True, x=0.1, y=0.3, age_ms=0.0)
    check("position/goal left pushes target left", desired_ball_x(ball, goal_left, c) < 0.5)

    # The offset must stay bounded however extreme the geometry.
    far = TrackedTarget(visible=True, confirmed=True, x=1.0, y=0.3, age_ms=0.0)
    edge = TrackedTarget(visible=True, confirmed=True, x=0.0, y=0.8, age_ms=0.0)
    offset = abs(desired_ball_x(edge, far, c) - 0.5)
    check("position/offset clamped", offset <= c.get_behind_max_offset + 1e-9, f"offset={offset}")


# ----------------------------------------------------- state machine flow
def test_state_machine() -> None:
    # Field-side awareness off: these cover state-machine mechanics, which are
    # orthogonal to which goal we are aimed at. The own-goal guard has its own
    # dedicated tests in test_own_goal_guard.
    c = cfg(detection_confirm_frames=1, align_stable_frames=1, attack_stable_frames=1,
            enable_field_side=False)

    # No ball -> SEARCH, and it must actually rotate.
    ctrl = SoccerController(c)
    command = ctrl.step(observation(0.0, ball_x=None, age_ms=float("inf")), 0.1)
    check("fsm/search when no ball", ctrl.state is State.SEARCH)
    check("fsm/search rotates", command.left != 0.0 or command.right != 0.0)

    # Confirmed ball -> ALIGN.
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=None, age_ms=float("inf")), 0.1)
    ctrl.step(observation(100.0, ball_x=0.3, ball_y=0.2), 0.1)
    check("fsm/acquire enters align", ctrl.state is State.ALIGN, f"state={ctrl.state}")

    # Centred ball -> CHASE.
    ctrl.step(observation(200.0, ball_x=0.5, ball_y=0.2), 0.1)
    check("fsm/centered enters chase", ctrl.state is State.CHASE, f"state={ctrl.state}")

    # Close and centred -> ATTACK.
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.5, ball_y=0.9), 0.1)
    ctrl.step(observation(100.0, ball_x=0.5, ball_y=0.9), 0.1)
    check("fsm/close centered attacks", ctrl.state is State.ATTACK, f"state={ctrl.state}")

    # The burst must survive a frame with no ball at all.
    command = ctrl.step(observation(150.0, ball_x=None, age_ms=50.0), 0.05)
    check("fsm/attack survives missing frame", ctrl.state is State.ATTACK)
    check("fsm/attack drives forward", command.left > 0 and command.right > 0)

    # ...but must end on its own deadline.
    ctrl.step(observation(150.0 + c.attack_burst_ms + 10, ball_x=None, age_ms=500.0), 0.1)
    check("fsm/attack ends on deadline", ctrl.state is State.RECOVER, f"state={ctrl.state}")

    # Long loss -> RECOVER -> SEARCH after timeout.
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.5, ball_y=0.2), 0.1)
    ctrl.step(observation(100.0, ball_x=None, age_ms=c.lost_grace_ms + 50), 0.1)
    check("fsm/loss enters recover", ctrl.state is State.RECOVER, f"state={ctrl.state}")
    ctrl.step(
        observation(100.0 + c.recover_timeout_ms + 10, ball_x=None, age_ms=99999.0), 0.1
    )
    check("fsm/recover times out to search", ctrl.state is State.SEARCH, f"state={ctrl.state}")

    # Reacquisition returns to ALIGN.
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.5, ball_y=0.2), 0.1)
    ctrl.step(observation(100.0, ball_x=None, age_ms=c.lost_grace_ms + 50), 0.1)
    # Off-centre so it stays in ALIGN; a centred ball would advance straight
    # to CHASE in the same tick, which is also correct behaviour.
    ctrl.step(observation(200.0, ball_x=0.2, ball_y=0.2), 0.1)
    check("fsm/reacquire enters align", ctrl.state is State.ALIGN, f"state={ctrl.state}")

    # A short dropout must not thrash the state machine.
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.5, ball_y=0.2), 0.1)
    ctrl.step(observation(100.0, ball_x=0.5, ball_y=0.2), 0.1)
    before = ctrl.state
    ctrl.step(observation(150.0, ball_x=None, age_ms=50.0), 0.05)
    check("fsm/short dropout keeps state", ctrl.state is before, f"{before} -> {ctrl.state}")

    # Unhealthy vision stops instantly, regardless of state.
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.5, ball_y=0.2), 0.1)
    command = ctrl.step(observation(100.0, ball_x=0.5, ball_y=0.2, healthy=False), 0.1)
    check("fsm/unhealthy stops motors", command.is_stop(), f"cmd={command}")

    # Turn direction must match which side the ball is on.
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.9, ball_y=0.2), 0.1)
    command = ctrl.step(observation(100.0, ball_x=0.9, ball_y=0.2), 0.1)
    check("fsm/ball right turns right", command.left > command.right, f"cmd={command}")

    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.1, ball_y=0.2), 0.1)
    command = ctrl.step(observation(100.0, ball_x=0.1, ball_y=0.2), 0.1)
    check("fsm/ball left turns left", command.right > command.left, f"cmd={command}")

    # Cooldown must prevent immediately re-attacking.
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.5, ball_y=0.9), 0.1)
    ctrl.step(observation(50.0, ball_x=0.5, ball_y=0.9), 0.1)
    ctrl.step(observation(50.0 + c.attack_burst_ms + 1, ball_x=0.5, ball_y=0.9), 0.1)
    ctrl.step(observation(50.0 + c.attack_burst_ms + 20, ball_x=0.5, ball_y=0.9), 0.1)
    check("fsm/cooldown blocks re-attack", ctrl.state is not State.ATTACK, f"state={ctrl.state}")


def test_contradictory_guards_do_not_recurse() -> None:
    """Regression: a target both confirmed AND stale must not bounce forever.

    The frame-repeat path in VisionAdapter preserves `confirmed` while age
    grows, so ALIGN saw "lost" and RECOVER saw "confirmed" and the two state
    bodies called each other until the stack blew.
    """
    c = cfg(detection_confirm_frames=1, align_stable_frames=1, attack_stable_frames=1)
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.3, ball_y=0.2), 0.1)

    # Confirmed, but older than the grace period: the contradictory case.
    stale_but_confirmed = observation(
        100.0, ball_x=0.3, ball_y=0.2, confirmed=True, age_ms=c.lost_grace_ms + 200
    )
    try:
        command = ctrl.step(stale_but_confirmed, 0.1)
        check("fsm/contradictory guards terminate", True)
        check("fsm/contradictory guards give a command", isinstance(command, MotorCommand))
        check(
            "fsm/contradictory guards bounded",
            abs(command.left) <= c.max_motor_command,
            f"cmd={command}",
        )
    except RecursionError:
        check("fsm/contradictory guards terminate", False, "RecursionError")

    # A ball still present in the last real frame must NOT be called lost just
    # because the camera stalled: that thrashed CHASE <-> RECOVER on hardware.
    # A stalled camera is the health check's job, not the tracker's.
    transitions_before = len(ctrl.transitions)
    for i in range(20):
        ctrl.step(
            observation(
                200.0 + i * 100,
                ball_x=0.3,
                ball_y=0.2,
                confirmed=True,
                age_ms=c.lost_grace_ms + 200,
            ),
            0.1,
        )
    churn = len(ctrl.transitions) - transitions_before
    check("fsm/visible stale ball does not thrash", churn <= 2, f"{churn} transitions")
    check("fsm/visible stale ball keeps pursuing",
          ctrl.state in (State.ALIGN, State.CHASE), f"state={ctrl.state}")

    # A genuinely absent ball must still reach RECOVER and then SEARCH.
    ctrl2 = SoccerController(c)
    ctrl2.step(observation(0.0, ball_x=0.3, ball_y=0.2), 0.1)
    for i in range(6):
        ctrl2.step(
            observation(100.0 + i * 200, ball_x=None, age_ms=c.lost_grace_ms + 200), 0.1
        )
    check("fsm/absent ball reaches recover", ctrl2.state is State.RECOVER,
          f"state={ctrl2.state}")


def test_one_state_body_per_iteration() -> None:
    """The spec requires it, and it is what prevents mutual recursion."""
    import inspect

    from soccer import controller as controller_module

    source = inspect.getsource(controller_module.SoccerController)
    body_start = source.find("def _command_for_state")
    bodies = source[body_start:]
    # No state body may call another state body.
    for caller in ("_search", "_align", "_chase", "_attack", "_recover"):
        marker = f"def {caller}(self"
        index = bodies.find(marker)
        if index == -1:
            continue
        nxt = min(
            (bodies.find(f"def {other}(self", index + 1)
             for other in ("_search", "_align", "_chase", "_attack", "_recover")
             if bodies.find(f"def {other}(self", index + 1) != -1),
            default=len(bodies),
        )
        body = bodies[index:nxt]
        for other in ("_search", "_align", "_chase", "_attack", "_recover"):
            if other == caller:
                continue
            check(
                f"fsm/{caller} does not call {other}",
                f"self.{other}(" not in body,
                f"{caller} calls {other}",
            )


def test_step_and_scan_search() -> None:
    """Search must alternate turning with holding still for a clean frame."""
    c = cfg(detection_confirm_frames=1, search_pulse_ms=200.0, search_settle_ms=400.0)
    ctrl = SoccerController(c)

    turning = scanning = 0
    for i in range(30):
        command = ctrl.step(observation(i * 100.0, ball_x=None, age_ms=99999.0), 0.1)
        if command.is_stop():
            scanning += 1
        else:
            turning += 1

    check("search/turns sometimes", turning > 0, f"turning={turning}")
    check("search/holds still sometimes", scanning > 0, f"scanning={scanning}")
    # 200ms turn / 400ms settle means it should be still more often than not.
    check("search/settles more than it turns", scanning > turning,
          f"turn={turning} scan={scanning}")

    # Disabling the settle window must restore continuous rotation.
    c2 = cfg(search_pulse_ms=200.0, search_settle_ms=0.0)
    ctrl2 = SoccerController(c2)
    stops = sum(
        1
        for i in range(10)
        if ctrl2.step(observation(i * 100.0, ball_x=None, age_ms=99999.0), 0.1).is_stop()
    )
    check("search/settle can be disabled", stops == 0, f"stops={stops}")


def test_own_goal_guard() -> None:
    """The bug that cost the final: shooting into our own net.

    RED team defends the RED wall, so facing RED must never fire.
    """
    c = cfg(
        detection_confirm_frames=1, align_stable_frames=1, attack_stable_frames=1,
        enable_field_side=True, own_wall_is_team_colour=True,
    )

    def close_ball(t, facing, blue=False):
        return observation(t, ball_x=0.5, ball_y=0.9, facing_side=facing, team_is_blue=blue)

    # RED team facing the RED wall = our own end. Must NOT attack.
    ctrl = SoccerController(c)
    ctrl.step(close_ball(0.0, "RED"), 0.1)
    ctrl.step(close_ball(100.0, "RED"), 0.1)
    check("own_goal/red team facing red does not attack", ctrl.state is not State.ATTACK,
          f"state={ctrl.state}")
    check("own_goal/red team facing red repositions", ctrl.state is State.REPOSITION,
          f"state={ctrl.state}")

    # RED team facing BLUE = their end. Must attack.
    ctrl = SoccerController(c)
    ctrl.step(close_ball(0.0, "BLUE"), 0.1)
    ctrl.step(close_ball(100.0, "BLUE"), 0.1)
    check("own_goal/red team facing blue attacks", ctrl.state is State.ATTACK,
          f"state={ctrl.state}")

    # BLUE team is the mirror image.
    ctrl = SoccerController(c)
    ctrl.step(close_ball(0.0, "BLUE", blue=True), 0.1)
    ctrl.step(close_ball(100.0, "BLUE", blue=True), 0.1)
    check("own_goal/blue team facing blue does not attack", ctrl.state is not State.ATTACK,
          f"state={ctrl.state}")

    ctrl = SoccerController(c)
    ctrl.step(close_ball(0.0, "RED", blue=True), 0.1)
    ctrl.step(close_ball(100.0, "RED", blue=True), 0.1)
    check("own_goal/blue team facing red attacks", ctrl.state is State.ATTACK,
          f"state={ctrl.state}")

    # Unknown side: no shot by default. A coin flip is what lost the final.
    ctrl = SoccerController(c)
    ctrl.step(close_ball(0.0, "UNKNOWN"), 0.1)
    ctrl.step(close_ball(100.0, "UNKNOWN"), 0.1)
    check("own_goal/unknown side does not attack", ctrl.state is not State.ATTACK,
          f"state={ctrl.state}")

    # ...unless explicitly allowed.
    c2 = cfg(detection_confirm_frames=1, align_stable_frames=1, attack_stable_frames=1,
             enable_field_side=True, attack_when_side_unknown=True)
    ctrl = SoccerController(c2)
    ctrl.step(close_ball(0.0, "UNKNOWN"), 0.1)
    ctrl.step(close_ball(100.0, "UNKNOWN"), 0.1)
    check("own_goal/unknown attacks when permitted", ctrl.state is State.ATTACK,
          f"state={ctrl.state}")

    # Inverted venue convention must flip the whole decision.
    c3 = cfg(detection_confirm_frames=1, align_stable_frames=1, attack_stable_frames=1,
             enable_field_side=True, own_wall_is_team_colour=False)
    ctrl = SoccerController(c3)
    ctrl.step(close_ball(0.0, "RED"), 0.1)
    ctrl.step(close_ball(100.0, "RED"), 0.1)
    check("own_goal/inverted convention attacks red", ctrl.state is State.ATTACK,
          f"state={ctrl.state}")

    # Feature off = old behaviour, shoots regardless.
    c4 = cfg(detection_confirm_frames=1, align_stable_frames=1, attack_stable_frames=1,
             enable_field_side=False)
    ctrl = SoccerController(c4)
    ctrl.step(close_ball(0.0, "RED"), 0.1)
    ctrl.step(close_ball(100.0, "RED"), 0.1)
    check("own_goal/disabled restores old behaviour", ctrl.state is State.ATTACK,
          f"state={ctrl.state}")


def test_reposition_behaviour() -> None:
    c = cfg(
        detection_confirm_frames=1, align_stable_frames=1, attack_stable_frames=1,
        enable_field_side=True, reposition_backoff_ms=400.0,
    )
    ctrl = SoccerController(c)
    ctrl.step(observation(0.0, ball_x=0.5, ball_y=0.9, facing_side="RED"), 0.1)
    cmd = ctrl.step(observation(100.0, ball_x=0.5, ball_y=0.9, facing_side="RED"), 0.1)
    check("reposition/entered", ctrl.state is State.REPOSITION, f"state={ctrl.state}")

    # Phase 1 backs away rather than turning while touching the ball.
    check("reposition/backs off first", cmd.left < 0 and cmd.right < 0, f"cmd={cmd}")

    # Phase 2 turns.
    cmd = ctrl.step(observation(600.0, ball_x=0.5, ball_y=0.9, facing_side="RED"), 0.1)
    check("reposition/then turns", cmd.left != cmd.right, f"cmd={cmd}")

    # Turn direction must not oscillate between frames.
    first = cmd.left > cmd.right
    for i in range(6):
        cmd = ctrl.step(
            observation(700.0 + i * 100, ball_x=0.5, ball_y=0.9, facing_side="RED"), 0.1
        )
    check("reposition/direction is stable", (cmd.left > cmd.right) == first, f"cmd={cmd}")

    # Once the far wall is in view, resume pursuit.
    ctrl.step(observation(1400.0, ball_x=0.5, ball_y=0.9, facing_side="BLUE"), 0.1)
    check("reposition/exits when turned", ctrl.state is not State.REPOSITION,
          f"state={ctrl.state}")

    # It must never spin forever.
    ctrl2 = SoccerController(c)
    ctrl2.step(observation(0.0, ball_x=0.5, ball_y=0.9, facing_side="RED"), 0.1)
    ctrl2.step(observation(100.0, ball_x=0.5, ball_y=0.9, facing_side="RED"), 0.1)
    ctrl2.step(
        observation(100.0 + c.reposition_timeout_ms + 50, ball_x=0.5, ball_y=0.9,
                    facing_side="RED"),
        0.1,
    )
    check("reposition/times out", ctrl2.state is not State.REPOSITION, f"state={ctrl2.state}")


def test_field_side_mapping() -> None:
    from soccer.field import Side, attack_side, facing_own_goal, own_side

    c = cfg(own_wall_is_team_colour=True)
    check("field/red team defends red", own_side(False, c) is Side.RED)
    check("field/red team attacks blue", attack_side(False, c) is Side.BLUE)
    check("field/blue team defends blue", own_side(True, c) is Side.BLUE)
    check("field/blue team attacks red", attack_side(True, c) is Side.RED)

    ci = cfg(own_wall_is_team_colour=False)
    check("field/inverted red team defends blue", own_side(False, ci) is Side.BLUE)

    check("field/facing own is detected", facing_own_goal(Side.RED, False, c))
    check("field/facing theirs is safe", not facing_own_goal(Side.BLUE, False, c))
    check("field/unknown is not own", not facing_own_goal(Side.UNKNOWN, False, c))


def test_goal_positioning_disabled_by_default() -> None:
    c = cfg(detection_confirm_frames=1, align_stable_frames=1, attack_stable_frames=1)
    check("fsm/goal detection off by default", c.enable_goal_detection is False)

    # With positioning off, a visible goal must not change the chase error.
    ctrl_a = SoccerController(c)
    ctrl_a.step(observation(0.0, ball_x=0.6, ball_y=0.5), 0.1)
    a = ctrl_a.step(observation(100.0, ball_x=0.6, ball_y=0.5), 0.1)

    ctrl_b = SoccerController(c)
    ctrl_b.step(observation(0.0, ball_x=0.6, ball_y=0.5, goal_x=0.9), 0.1)
    b = ctrl_b.step(observation(100.0, ball_x=0.6, ball_y=0.5, goal_x=0.9), 0.1)
    check("fsm/goal ignored when disabled", abs(a.left - b.left) < 1e-9)


# ----------------------------------------------------------- motor adapter
def test_motor_adapter() -> None:
    c = cfg()
    bridge = FakeBridge()
    motors = MotorAdapter(c, bridge)

    motors.apply(MotorCommand(1.0, 1.0, "fwd"))
    call = bridge.last_drive
    check("motor/forward all positive", all(v > 0 for v in call[1:5]), f"call={call}")
    check("motor/watchdog attached", call[5] == c.command_watchdog_ms, f"call={call}")

    # Turning right: left side (m1, m2) faster than right side (m0, m3).
    bridge.calls.clear()
    motors.apply(MotorCommand(0.8, 0.2, "right"))
    _, m0, m1, m2, m3, _ = bridge.last_drive
    check("motor/side mapping", m1 == m2 and m0 == m3, f"m={m0},{m1},{m2},{m3}")
    check("motor/turn right left faster", m1 > m0, f"left={m1} right={m0}")

    bridge.calls.clear()
    motors.apply(MotorCommand.stop("halt"))
    check("motor/stop is zeros", bridge.last_drive[1:5] == (0, 0, 0, 0), f"{bridge.last_drive}")

    bridge.calls.clear()
    motors.apply(MotorCommand(9.0, -9.0, "over"))
    _, m0, m1, m2, m3, _ = bridge.last_drive
    check("motor/clamped to 255", max(abs(m0), abs(m1)) <= 255, f"m={m0},{m1}")

    # Below the stiction floor the command must be scaled up, not sent as-is.
    c2 = cfg(min_effective_speed=0.5)
    bridge = FakeBridge()
    motors = MotorAdapter(c2, bridge)
    motors.apply(MotorCommand(0.1, 0.1, "tiny"))
    _, m0, m1, m2, m3, _ = bridge.last_drive
    check("motor/raises below floor", abs(m0) >= int(0.5 * 255) - 1, f"m0={m0}")

    # ...and raising it must not change the turn ratio.
    bridge.calls.clear()
    motors.apply(MotorCommand(0.2, 0.1, "tiny turn"))
    _, m0, m1, m2, m3, _ = bridge.last_drive
    check("motor/floor preserves turn ratio", abs(m1 / m0 - 2.0) < 0.05, f"m0={m0} m1={m1}")

    # A floor scale-up must never exceed the output limit.
    c3 = cfg(min_effective_speed=0.9, max_motor_command=1.0)
    bridge = FakeBridge()
    motors = MotorAdapter(c3, bridge)
    motors.apply(MotorCommand(0.05, -0.05, "tiny opposite"))
    _, m0, m1, m2, m3, _ = bridge.last_drive
    check("motor/floor respects limit", max(abs(m0), abs(m1)) <= 255, f"m={m0},{m1}")

    # Zero must stay zero: the floor must not make a stop start moving.
    bridge.calls.clear()
    motors.apply(MotorCommand.stop("halt"))
    check("motor/floor never revives a stop", bridge.last_drive[1:5] == (0, 0, 0, 0))


def main() -> int:
    for fn in (
        test_config,
        test_steering,
        test_safety,
        test_tracker,
        test_memory,
        test_positioning,
        test_state_machine,
        test_contradictory_guards_do_not_recurse,
        test_step_and_scan_search,
        test_own_goal_guard,
        test_reposition_behaviour,
        test_field_side_mapping,
        test_one_state_body_per_iteration,
        test_goal_positioning_disabled_by_default,
        test_motor_adapter,
    ):
        fn()

    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    for failure in FAILED:
        print(f"  FAIL {failure}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
