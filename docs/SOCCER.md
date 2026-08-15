# Autonomous Soccer Controller

Implementation of `AUTONOMOUS_SOCCER_SPEC_CONCISE.md` for the Hiwonder miniAuto
on Arduino UNO Q.

## Layout

```text
python/
  soccer_main.py          entry point + debug modes
  soccer/
    config.py             every tunable, with validation
    types.py              Detection / TrackedTarget / WorldObservation / MotorCommand
    tracker.py            streak gating, filtering, last-seen memory
    steering.py           proportional steering, speed staging
    positioning.py        goal-relative "get behind the ball"
    field.py              which end are we facing (own-goal guard)
    safety.py             clamp, slew limit, emergency stop, watchdog
    controller.py         SEARCH / ALIGN / CHASE / ATTACK / RECOVER / REPOSITION
    adapters.py           motor + vision adapters, and fakes for tests
    telemetry.py          rate-limited structured output
  tests/test_soccer.py    115 off-robot tests, no dependencies
```

## Run it

```bash
adb push python/soccer python/vision.py python/main.py python/soccer_main.py \
  /home/arduino/ArduinoApps/miniautodriver/python/
adb shell 'rm -rf /home/arduino/ArduinoApps/miniautodriver/python/soccer/__pycache__'
adb shell 'arduino-app-cli app restart user:miniautodriver'
adb shell 'arduino-app-cli app logs user:miniautodriver | tail -30'
```

Tests run anywhere, no hardware and no pytest:

```bash
python3 python/tests/test_soccer.py
```

## Modes

Set `ROBOCUP_MODE`:

| Mode | Behaviour |
| --- | --- |
| `VISION_ONLY` | Full control loop, motors forced to zero. Safe anywhere. |
| `FIELD_SIDE_CHECK` | Confirm which wall colour means which goal. No motion. |
| `MOTORS_ON_BLOCKS` | Four low-power moves to verify polarity. Wheels raised. |
| `MOTORS_RAMP` | Find the minimum speed that turns the wheels. Wheels raised. |
| `FIELD_RUN` (default) | Live control. Requires arming. |

## Arming and emergency stop

Arming is the **CAM BOOT button**, read through the firmware's
`program_enabled` flag. The robot cannot drive while disarmed, and pressing the
button again disarms and stops it. `ROBOCUP_MODE` other than `FIELD_RUN` forces
motors to zero regardless of arming.

Three independent stop paths:

1. **Button** — disarms; the loop commands zero.
2. **Software** — any exception, non-finite command, unhealthy vision or
   disarm produces an immediate stop that bypasses slew limiting.
3. **Firmware** — every command carries `command_watchdog_ms` as the drive
   timer, so if this process dies mid-drive the MCU stops the wheels itself.

## Measured values

Established on this robot, not guesses:

| Value | Measurement |
| --- | --- |
| Camera | 5.75 FPS; gaps median 98 ms, p90 227 ms, max 1792 ms |
| Inference | 25 ms steady state (~40 FPS capability) |
| Model | FOMO 96x96, labels `goal`, `robot`, `soccer_ball` |
| Runner threshold | 0.15 — at the shipped 0.5 the ball only fires near the lens |
| Minimum wheel speed | wheels turn from the 0.20–0.30 band, floor set to 0.25 |
| Battery (healthy) | ~12000 mV; a flat pack reads ~4900 mV and breaks everything |

`vision_stale_ms=2400` and `lost_grace_ms=700` come from the camera's measured
tail, not from the spec's defaults. The 1.8 s outlier is present even while
parked, so it is RF congestion (many competing team APs on 2.4 GHz), not motion.
At 1500 ms roughly 21% of frames tripped the health check and the robot spent
its time stopping instead of playing; at 2400 ms that fell to 3%.

The control loop runs at 10 Hz — no value in going faster than the camera.

## Field-side awareness (the own-goal fix)

**Why this exists:** the robot placed second in the tournament, losing the final
to two own goals. With no notion of direction, `ATTACK` pushed the ball whichever
way the robot happened to face — a coin flip on every shot.

Retraining the model cannot fix this. The two goals are physically identical, so
no classifier can separate them. The only signal that differs between the ends is
the red/blue wall tape.

`soccer/field.py` classifies the wall behind the ball by hue and uses it to decide
whether a forward push goes toward their net or ours.

### Verify the mapping before you play

```bash
ROBOCUP_MODE=FIELD_SIDE_CHECK   # no motion; point the robot at each end
```

It prints, per end:

```
red=7.2% blue=0.1%   stable=RED    OWN GOAL - would not shoot
red=0.2% blue=6.8%   stable=BLUE   THEIR GOAL - would shoot
```

`own_wall_is_team_colour` defaults to true, meaning a RED team defends the RED end.
**If your tournament tapes the end you attack instead, set
`ROBOCUP_OWN_WALL_IS_TEAM=false`** — backwards, this doubles own goals rather than
preventing them. This is the single most important thing to check at a venue.

### How it behaves

| Situation | Action |
| --- | --- |
| Facing their end | Attack normally |
| Facing our end, ball close | `REPOSITION` — back off, then arc around the ball |
| Side unknown | Do not shoot (`ROBOCUP_ATTACK_IF_UNKNOWN=true` to override) |

Gating `ATTACK` alone is not enough: `CHASE` also drives forward into the ball, so
the robot must change its *approach*, not just withhold the shot. And rotating in
place is the wrong manoeuvre — to push a ball the other way you have to reach the
opposite side of it, so `REPOSITION` arcs around it. The exit needs no odometry:
circling makes the wall behind the ball change colour by itself.

Disable the whole feature with `ROBOCUP_ENABLE_FIELD_SIDE=false`.

**Status: implemented and unit-tested, never run on hardware.** The HSV thresholds
in particular need real tape under real lighting — run `FIELD_SIDE_CHECK` first.

## Two deliberate deviations from the spec

**1. Goal positioning is disabled by default.** The model has a single `goal`
class and cannot distinguish our net from theirs. The spec itself says not to
guess the attacking goal without a distinguishing rule. Enable with
`ROBOCUP_ENABLE_GOAL=true` only once you have such a rule, or you risk own
goals.

**2. The get-behind sign is flipped versus the written formula.** The spec says
`desiredBallX = 0.5 - clamp(...)` but also says a goal right of the ball means
approaching the ball's left side. Those contradict: with the minus sign the
robot steers to the ball's right and pushes the ball away from the goal.
`positioning.py` uses `0.5 + clamp(...)`, which matches the prose and the
physics. Covered by `test_positioning`.

## Hardware mapping

The spec assumes differential drive; this chassis is mecanum. From
`velocityController(angle=0)` in `sketch.ino` the wheels collapse into two
virtual sides:

```text
m0, m3 = right side      m1, m2 = left side
left  = base + turn      right  = base - turn
```

`drive_raw` is used rather than the named `drive` commands because it bypasses
`velocityController`'s `1/sqrt(2)` scaling and its `speedFactor=0.5` rotation
penalty, giving the controller full authority and a single place to clamp.

## Camera network

The ESP32-S3 camera is its own access point. `wlan0` must be joined to it:

```bash
nmcli device wifi connect miniAuto_CAM_30 password 'Q30pass!'
```

The SSID number matches the robot; several teams' cameras are usually visible.
Joining it means losing internet, so install anything you need first.

## Limitations

- Detection range is bounded by FOMO at 96x96; the ball must occupy enough
  pixels. Lower `ROBOCUP_RUNNER_THRESHOLD` for more range at the cost of false
  positives.
- No opponent avoidance yet (`ROBOCUP_ENABLE_OPPONENT`, default off).
- No kicker; the shot is a timed forward burst.
- `ball.y` is a coarse distance proxy and needs calibrating on the real field:
  record `ball.y` at far/mid/contact and set `ball_y_far` / `ball_y_close`
  between the distributions.
- Field-side awareness is implemented but never run on hardware. Verify with
  `FIELD_SIDE_CHECK` before trusting it.
- `ball_y_far` / `ball_y_close` were never calibrated on a real field; the robot
  scored with the spec's defaults, but approach speed is the knob if it
  overshoots or crawls.

## Operating notes learned the hard way

- **Check `battery_mv` before any other theory.** A flat pack presents as motors
  dead, camera AP vanished, and ultrasonic returning 65535 — all at once, while
  adb and inference keep working on USB power. It cost hours twice.
- **Do not restart the app to recover the camera.** The ESP32 serves one client;
  a restart can leave it refusing connections. Power cycle instead.
- **`ARMED` lives in the MCU, not the app.** It survives app restarts and stops.
  Only the CAM button or a power cycle clears it.
- **A Bridge `TimeoutError` is only fixed by a full power cycle.** `Bridge.begin()`
  runs once in `setup()` with no reconnect, so app restarts and reflashes do not
  recover the link.
