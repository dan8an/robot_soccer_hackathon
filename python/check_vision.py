"""Vision bring-up check. Loads the model, reads the camera, prints detections.

Moves nothing - run this before ever letting the robot play. Copy over
python/main.py, start the app, and read the logs.
"""
import time

from vision import CameraStream, Detector, find_model

RUN_SECONDS = 40.0

print("=" * 60)
print("VISION CHECK - no motion")
print("=" * 60)

try:
    model_path = find_model()
    print(f"[1/3] model file : {model_path.name} ({model_path.stat().st_size // 1024} KB)")
except FileNotFoundError as exc:
    print(f"[1/3] FAIL - {exc}")
    raise SystemExit(1)

camera = CameraStream()
camera.start()

print("[2/3] waiting for camera frames...")
deadline = time.monotonic() + 15.0
while not camera.is_fresh() and time.monotonic() < deadline:
    time.sleep(0.25)

frame, age = camera.latest()
if frame is None:
    print("[2/3] FAIL - no frames. Check wlan0 is joined to the camera AP")
    print("      nmcli device wifi connect miniAuto_CAM_01")
    camera.stop()
    raise SystemExit(1)
print(f"[2/3] camera OK - frame {frame.size[0]}x{frame.size[1]}, age {age:.2f}s")

print("[3/3] running inference for 40 s - hold objects in front of the camera")
try:
    with Detector() as detector:
        expected = {"soccer_ball", "goal", "robot"}
        if not expected & set(detector.labels):
            print(
                f"      NOTE: none of {sorted(expected)} are in this model's labels. "
                "The soccer model is probably not selected for the brick."
            )
        counts: dict[str, int] = {}
        frames = 0
        end = time.monotonic() + RUN_SECONDS
        while time.monotonic() < end:
            frame, age = camera.latest()
            if frame is None or age > 1.5:
                time.sleep(0.1)
                continue
            started = time.monotonic()
            detections = detector.detect(frame)
            elapsed_ms = (time.monotonic() - started) * 1000
            frames += 1
            for d in detections:
                counts[d.label] = counts.get(d.label, 0) + 1
            summary = ", ".join(
                f"{d.label}@{d.confidence:.2f} bearing={d.bearing:+.2f}" for d in detections
            )
            print(f"  [{elapsed_ms:5.0f} ms] {summary or '(nothing detected)'}")
            time.sleep(0.1)

        print("-" * 60)
        print(f"frames inferred : {frames}")
        print(f"labels seen     : {sorted(counts) or '(none - model saw nothing)'}")
        for label, seen in sorted(counts.items()):
            print(f"  {label:<14} detected in {seen} frames")
        print("VISION CHECK complete - nothing was driven.")
finally:
    camera.stop()
