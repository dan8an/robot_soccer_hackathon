"""MATCH ENTRY POINT - autonomous soccer.

This is what the board runs on boot. It starts DISARMED and waits for the CAM
BOOT button, so powering on never makes the robot move by itself.

  Battery on -> app autostarts -> press CAM BOOT -> red/amber/green -> plays
  Press CAM BOOT again -> stops

To run a different mode without editing this file, set ROBOCUP_MODE before the
app starts:
  FIELD_SIDE_CHECK  confirm which wall colour means which goal (no motion)
  VISION_ONLY       inference and telemetry, motors forced to zero
  MOTORS_ON_BLOCKS  polarity check, wheels raised
  MOTORS_RAMP       find the minimum speed that turns the wheels

The original driver demo is preserved in demo_main.py.
"""
import os
import runpy

# FIELD_RUN is the match default. An explicit ROBOCUP_MODE still wins, so the
# diagnostic modes remain reachable without touching this file.
os.environ.setdefault("ROBOCUP_MODE", "FIELD_RUN")

try:
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "soccer_main.py"), run_name="__main__"
    )
finally:
    # soccer_main.py stops the motors in its own finally block. This is a
    # second, independent net: if anything at all escapes - an import error, a
    # failure inside runpy, an exception during its shutdown path - the wheels
    # must still be commanded to zero. A robot that keeps driving after its
    # controller has died is the worst failure this code can have.
    #
    # robot.stop() rather than a raw motor zero, because it also DISARMS the
    # firmware. A crashed controller that leaves the program armed will start
    # driving again the moment vision recovers.
    try:
        from robot_client import MiniAutoRobot

        robot = MiniAutoRobot()
        robot.stop()
    except Exception:  # noqa: BLE001 - already failing; never mask the original
        pass
