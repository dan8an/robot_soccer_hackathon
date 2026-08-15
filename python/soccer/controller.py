"""The SEARCH / ALIGN / CHASE / ATTACK / RECOVER state machine.

Pure logic: it takes a WorldObservation and returns a MotorCommand. It touches
no hardware, which is what lets the whole strategy be tested off-robot.
"""
from . import field, positioning
from .config import Config
from .field import Side
from .steering import SteeringController, clamp, speed_from_ball_y
from .tracker import BallMemory
from .types import MotorCommand, State, TrackedTarget, WorldObservation


class SoccerController:
    def __init__(self, config: Config, memory: BallMemory | None = None) -> None:
        self._config = config
        self._steering = SteeringController(config)
        self.memory = memory if memory is not None else BallMemory(config)

        self.state = State.SEARCH
        self.state_entered_ms = 0.0
        self.last_transition_reason = "startup"
        self.transitions: list[tuple[float, str, str, str]] = []

        self._align_stable = 0
        self._attack_stable = 0
        self._search_flip_ms = 0.0
        self._search_direction = config.default_search_direction
        self._attack_deadline_ms = 0.0
        self._attack_correction = 0.0
        self._attack_finished_ms = -1e9
        self._reposition_direction = 1
        self._reposition_finished_ms = -1e9
        self._positioning_started_ms: float | None = None
        self._last_command = MotorCommand.stop("startup")

    # ------------------------------------------------------------------
    # transitions
    # ------------------------------------------------------------------
    def _transition(
        self,
        new_state: State,
        now_ms: float,
        reason: str,
        obs: WorldObservation | None = None,
    ) -> None:
        if new_state is self.state:
            return
        self.transitions.append((now_ms, self.state.value, new_state.value, reason))
        self.state = new_state
        self.state_entered_ms = now_ms
        self.last_transition_reason = reason
        self._align_stable = 0
        self._attack_stable = 0
        self._steering.reset()
        if new_state is State.SEARCH:
            self._search_flip_ms = now_ms
        if new_state is not State.CHASE:
            self._positioning_started_ms = None
        if new_state is State.REPOSITION:
            # Latch a turn direction so the manoeuvre cannot oscillate. Turn
            # away from whichever side the ball sits on, so we swing around it
            # rather than into it.
            self._reposition_direction = -1 if obs is not None and obs.ball.x > 0.5 else 1
        if new_state is State.ATTACK:
            # Latch the deadline and correction on entry: the burst must
            # survive a missing frame, so it cannot depend on live vision.
            cfg = self._config
            self._attack_deadline_ms = now_ms + cfg.attack_burst_ms
            ball_x = obs.ball.x if obs is not None else 0.5
            self._attack_correction = clamp(
                cfg.kp_steer * (ball_x - 0.5), -cfg.max_turn, cfg.max_turn
            )

    def _time_in_state(self, now_ms: float) -> float:
        return now_ms - self.state_entered_ms

    # ------------------------------------------------------------------
    # readiness helpers
    # ------------------------------------------------------------------
    def _ball_lost_beyond_grace(self, ball: TrackedTarget) -> bool:
        """Lost means recent frames had no ball - not that frames stopped arriving.

        A stalled camera is a separate concern handled by the health check. If
        the most recent frame we actually looked at contained the ball, we have
        not lost it, however old that frame is. Conflating the two made a
        perfectly tracked ball (confidence 1.00) trigger ball_lost whenever the
        camera hiccuped, thrashing CHASE <-> RECOVER.
        """
        if ball.visible:
            return False
        return ball.age_ms > self._config.lost_grace_ms

    def _shooting_at_own_goal(self, obs: WorldObservation) -> bool:
        """Would a forward push send the ball into our own net?"""
        if not self._config.enable_field_side:
            return False
        return field.facing_own_goal(
            Side(obs.facing_side), obs.team_is_blue, self._config
        )

    def _shot_direction_allowed(self, obs: WorldObservation) -> bool:
        """Is it safe to fire the attack burst from here?"""
        cfg = self._config
        if not cfg.enable_field_side:
            return True
        facing = Side(obs.facing_side)
        if facing is Side.UNKNOWN:
            # No shot is better than a coin-flip own goal.
            return cfg.attack_when_side_unknown
        return not field.facing_own_goal(facing, obs.team_is_blue, cfg)

    def _ball_is_close(self, obs: WorldObservation) -> bool:
        cfg = self._config
        if obs.ball.y >= cfg.ball_y_close:
            return True
        return cfg.contact_cm > 0 and 0 < obs.ultrasonic_cm <= cfg.contact_cm

    def _attack_ready(self, obs: WorldObservation) -> bool:
        """Close enough, centred enough, and (optionally) aimed at the goal."""
        cfg = self._config
        ball = obs.ball
        if not ball.is_fresh(cfg.vision_stale_ms):
            return False

        # Never fire toward our own end. This is the own-goal guard.
        if not self._shot_direction_allowed(obs):
            return False

        close = ball.y >= cfg.ball_y_close
        if cfg.contact_cm > 0 and 0 < obs.ultrasonic_cm <= cfg.contact_cm:
            close = True
        if not close:
            return False

        if abs(ball.x - 0.5) > cfg.attack_ball_tolerance:
            return False

        if positioning.should_position(ball, obs.goal, cfg):
            # Only shoot once the ball sits where pushing sends it goalward.
            if abs(positioning.positioning_error(ball, obs.goal, cfg)) > cfg.attack_ball_tolerance:
                return False
            if abs(obs.goal.x - 0.5) > cfg.attack_goal_tolerance:
                return False
        return True

    def _reposition_cooling(self, now_ms: float) -> bool:
        """After a failed reposition, stop retrying it for a while.

        Without this the timeout is useless: the condition that triggered
        the manoeuvre is still true, so it would re-enter immediately.
        """
        return (now_ms - self._reposition_finished_ms) < self._config.reposition_cooldown_ms

    def _cooldown_active(self, now_ms: float) -> bool:
        return (now_ms - self._attack_finished_ms) < self._config.attack_cooldown_ms

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def step(self, obs: WorldObservation, dt_s: float) -> MotorCommand:
        now_ms = obs.timestamp_ms
        self.memory.update(obs.ball, now_ms)

        # System health is not ordinary ball loss: stop now, do not coast.
        if not obs.healthy:
            self._transition(State.RECOVER, now_ms, f"unhealthy:{obs.health_reason}")
            self._last_command = MotorCommand.stop(f"unhealthy:{obs.health_reason}")
            return self._last_command

        # Transitions are evaluated first and separately from command
        # generation. State bodies must never call one another: contradictory
        # guards (a target both confirmed and stale) would otherwise recurse
        # forever, and the spec requires one state body per iteration.
        self._evaluate_transitions(obs, now_ms)

        command = self._command_for_state(obs, now_ms, dt_s)
        self._last_command = command
        return command

    def _evaluate_transitions(self, obs: WorldObservation, now_ms: float) -> None:
        """Apply transitions until the state settles. Bounded, never recursive."""
        for _ in range(4):
            nxt = self._next_state(obs, now_ms)
            if nxt is None:
                return
            new_state, reason = nxt
            if new_state is self.state:
                return
            self._transition(new_state, now_ms, reason, obs)
        # A cycle would mean two guards contradict each other. Log it rather
        # than spinning; the current state still produces a bounded command.
        self.last_transition_reason = "transition_guard_tripped"

    def _next_state(self, obs: WorldObservation, now_ms: float) -> tuple[State, str] | None:
        cfg = self._config
        ball = obs.ball
        # Reacquisition requires a FRESH confirmation. Using confirmed alone
        # lets a stale-but-confirmed target bounce ALIGN <-> RECOVER.
        fresh_ball = ball.confirmed and ball.visible

        if self.state is State.SEARCH:
            if fresh_ball:
                return State.ALIGN, "ball_confirmed"
            return None

        if self.state is State.ALIGN:
            if self._ball_lost_beyond_grace(ball):
                return State.RECOVER, "ball_lost"
            if (self._ball_is_close(obs) and self._shooting_at_own_goal(obs)
                    and not self._reposition_cooling(now_ms)):
                return State.REPOSITION, "facing_own_goal"
            if self._attack_ready(obs) and not self._cooldown_active(now_ms):
                self._attack_stable += 1
                if self._attack_stable >= cfg.attack_stable_frames:
                    return State.ATTACK, "attack_ready"
            else:
                self._attack_stable = 0
            if abs(ball.x - 0.5) <= cfg.align_tolerance:
                self._align_stable += 1
                if self._align_stable >= cfg.align_stable_frames:
                    return State.CHASE, "centered"
            else:
                self._align_stable = 0
            return None

        if self.state is State.CHASE:
            if self._ball_lost_beyond_grace(ball):
                return State.RECOVER, "ball_lost"
            # Check this BEFORE attack readiness: chasing drives forward into
            # the ball, so continuing to chase while aimed at our own net
            # pushes it the wrong way even without firing a burst.
            if (self._ball_is_close(obs) and self._shooting_at_own_goal(obs)
                    and not self._reposition_cooling(now_ms)):
                return State.REPOSITION, "facing_own_goal"
            if self._attack_ready(obs) and not self._cooldown_active(now_ms):
                self._attack_stable += 1
                if self._attack_stable >= cfg.attack_stable_frames:
                    return State.ATTACK, "attack_ready"
            else:
                self._attack_stable = 0
            return None

        if self.state is State.REPOSITION:
            # Done as soon as we are no longer aimed at our own end.
            if not self._shooting_at_own_goal(obs):
                return State.ALIGN, "turned_to_attack_side"
            if self._time_in_state(now_ms) >= cfg.reposition_timeout_ms:
                # Give up on this approach entirely. Falling back to CHASE
                # would bounce straight back here, because the condition that
                # sent us here is still true.
                self._reposition_finished_ms = now_ms
                return State.RECOVER, "reposition_timeout"
            if not obs.ball.visible and self._ball_lost_beyond_grace(ball):
                return State.RECOVER, "ball_lost"
            return None

        if self.state is State.ATTACK:
            if now_ms >= self._attack_deadline_ms:
                self._attack_finished_ms = now_ms
                return State.RECOVER, "attack_burst_complete"
            return None

        # RECOVER
        if fresh_ball:
            return State.ALIGN, "ball_reacquired"
        if self._time_in_state(now_ms) >= cfg.recover_timeout_ms:
            return State.SEARCH, "recover_timeout"
        return None

    def _command_for_state(
        self, obs: WorldObservation, now_ms: float, dt_s: float
    ) -> MotorCommand:
        """Generate one command for the current state. No transition logic."""
        if self.state is State.SEARCH:
            return self._search(obs, now_ms)
        if self.state is State.ALIGN:
            return self._align(obs, now_ms, dt_s)
        if self.state is State.CHASE:
            return self._chase(obs, now_ms, dt_s)
        if self.state is State.ATTACK:
            return self._attack(obs, now_ms)
        if self.state is State.REPOSITION:
            return self._reposition(obs, now_ms)
        return self._recover(obs, now_ms)

    # ------------------------------------------------------------------
    # states
    # ------------------------------------------------------------------
    def _scanning(self, now_ms: float) -> bool:
        """True while holding still to let a clean frame arrive."""
        cfg = self._config
        cycle = cfg.search_pulse_ms + cfg.search_settle_ms
        if cycle <= 0:
            return False
        return ((now_ms - self.state_entered_ms) % cycle) >= cfg.search_pulse_ms

    def _search(self, obs: WorldObservation, now_ms: float) -> MotorCommand:
        cfg = self._config
        # Sweep back and forth so a ball behind us is eventually found.
        if (now_ms - self._search_flip_ms) >= cfg.search_sweep_ms:
            self._search_flip_ms = now_ms
            self._search_direction *= -1
        elif self._time_in_state(now_ms) < 1e-6:
            self._search_direction = self.memory.side(now_ms)

        # Step-and-scan: rotating continuously both starves the camera link and
        # sweeps past the ball between frames.
        if self._scanning(now_ms):
            return MotorCommand.stop("search_scan")

        return self._steering.rotate(self._search_direction, cfg.search_turn_speed, "search")

    def _align(self, obs: WorldObservation, now_ms: float, dt_s: float) -> MotorCommand:
        cfg = self._config
        return self._steering.command(
            cfg.align_base_speed, obs.ball.x - 0.5, dt_s, "align"
        )

    def _chase(self, obs: WorldObservation, now_ms: float, dt_s: float) -> MotorCommand:
        cfg = self._config
        ball = obs.ball
        base = speed_from_ball_y(ball.y, cfg)
        error = ball.x - 0.5
        reason = "chase"

        if positioning.should_position(ball, obs.goal, cfg):
            if self._positioning_started_ms is None:
                self._positioning_started_ms = now_ms
            elapsed = now_ms - self._positioning_started_ms
            if elapsed <= cfg.positioning_timeout_ms:
                # Aim for the offset approach line rather than the ball centre.
                error = positioning.positioning_error(ball, obs.goal, cfg)
                base = min(base, cfg.chase_speed_close)
                reason = "chase_get_behind"
            else:
                # Circling forever loses more than a straight push does.
                reason = "chase_positioning_timeout"
        else:
            self._positioning_started_ms = None

        return self._steering.command(base, error, dt_s, reason)

    def _attack(self, obs: WorldObservation, now_ms: float) -> MotorCommand:
        cfg = self._config
        limit = cfg.max_motor_command
        return MotorCommand(
            left=clamp(cfg.attack_speed + self._attack_correction, -limit, limit),
            right=clamp(cfg.attack_speed - self._attack_correction, -limit, limit),
            reason="attack",
        )

    def _reposition(self, obs: WorldObservation, now_ms: float) -> MotorCommand:
        """We are on the wrong side of the ball. Circle around to the other side.

        Rotating on the spot is not enough: to push the ball the other way we
        must physically get to the opposite side of it. So we back off just
        enough to stop shoving it, then drive an arc around it.

        The exit condition is self-terminating and needs no odometry - as we
        swing around the ball, the wall behind it changes colour by itself.
        """
        cfg = self._config
        elapsed = self._time_in_state(now_ms)

        # Backing off first matters: arcing while in contact drags the ball
        # along, which is how a robot nudges its own net anyway.
        if elapsed < cfg.reposition_backoff_ms:
            speed = cfg.reposition_speed
            return MotorCommand(-speed, -speed, "reposition_backoff")

        # Arc: both wheels forward but asymmetric, so we curve around the ball
        # rather than rotating in place or driving away from it.
        fast = cfg.reposition_speed
        slow = cfg.reposition_speed * cfg.reposition_arc_ratio
        if self._reposition_direction > 0:
            return MotorCommand(fast, slow, "reposition_arc")
        return MotorCommand(slow, fast, "reposition_arc")

    def _recover(self, obs: WorldObservation, now_ms: float) -> MotorCommand:
        cfg = self._config
        # Phase 1: coast briefly on the previous command at reduced power, so a
        # one-frame dropout does not cause a visible stutter.
        if obs.ball.age_ms <= cfg.lost_grace_ms and not self._last_command.is_stop():
            return MotorCommand(
                left=self._last_command.left * 0.5,
                right=self._last_command.right * 0.5,
                reason="recover_coast",
            )

        # Phase 2: turn back toward wherever the ball last was, in the same
        # step-and-scan rhythm so the camera can actually see it.
        if self._scanning(now_ms):
            return MotorCommand.stop("recover_scan")

        return self._steering.rotate(
            self.memory.side(now_ms), cfg.recover_turn_speed, "recover_turn"
        )
