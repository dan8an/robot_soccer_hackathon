"""Camera stream + Edge Impulse object detection for the miniAuto soccer robot.

The ESP32-S3 camera runs its own access point (see camera/HiwonderCamStream.ino),
so the UNO Q's wlan0 must be joined to CAMERA_SSID for frames to arrive.

Inference runs the Edge Impulse .eim directly: it is a self-contained aarch64
executable that speaks newline-delimited JSON over a Unix socket. Put the .eim
in models/ and it is picked up automatically.
"""
import io
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image

CAMERA_URL = os.environ.get("ROBOCUP_CAMERA_URL", "http://192.168.5.1:81/stream")
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_FOLDER / "models"

# A frame older than this means we have lost the camera; callers must stop.
STALE_FRAME_SECONDS = float(os.environ.get("ROBOCUP_STALE_SEC", "1.5"))


@dataclass
class Detection:
    """One FOMO centroid, in normalized frame coordinates."""

    label: str
    confidence: float
    x: float  # 0.0 = left edge, 1.0 = right edge
    y: float  # 0.0 = top edge, 1.0 = bottom edge

    @property
    def bearing(self) -> float:
        """Horizontal offset from frame center: -1.0 = hard left, +1.0 = hard right."""
        return (self.x - 0.5) * 2.0


class CameraStream:
    """Background MJPEG reader. Always hands back the most recent frame."""

    def __init__(self, url: str = CAMERA_URL) -> None:
        self._url = url
        self._frame: Image.Image | None = None
        self._frame_at = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[Image.Image | None, float]:
        """Return (frame, age_seconds). Frame is None until the first arrives."""
        frame, age, _ = self.latest_frame()
        return frame, age

    def latest_frame(self) -> tuple[Image.Image | None, float, int]:
        """Return (frame, age_seconds, sequence).

        The sequence number lets callers tell a genuinely new frame from the
        same one served again. That matters because this camera runs far
        slower than the control loop, and re-inferring one frame would
        otherwise inflate detection streaks.
        """
        with self._lock:
            if self._frame is None:
                return None, float("inf"), 0
            return self._frame.copy(), time.monotonic() - self._frame_at, self._seq

    def is_fresh(self) -> bool:
        _, age = self.latest()
        return age <= STALE_FRAME_SECONDS

    def _reader(self) -> None:
        while not self._stop.is_set():
            buffer = b""
            response = None
            try:
                print(f"[VISION] connecting to camera: {self._url}")
                response = requests.get(self._url, stream=True, timeout=(5, None))
                response.raise_for_status()
                print("[VISION] camera connected")
                for chunk in response.iter_content(chunk_size=4096):
                    if self._stop.is_set():
                        break
                    buffer += chunk
                    # MJPEG frames are delimited by JPEG SOI/EOI markers.
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", start + 2)
                        if start == -1 or end == -1:
                            break
                        jpg = buffer[start : end + 2]
                        buffer = buffer[end + 2 :]
                        try:
                            frame = Image.open(io.BytesIO(jpg)).convert("RGB")
                        except Exception as exc:  # noqa: BLE001 - corrupt frame, skip it
                            print(f"[VISION] bad frame: {exc}")
                            continue
                        with self._lock:
                            self._frame = frame
                            self._frame_at = time.monotonic()
                            self._seq += 1
            except requests.exceptions.RequestException as exc:
                print(f"[VISION] stream interrupted: {exc}")
            finally:
                if response is not None:
                    response.close()
            if not self._stop.is_set():
                time.sleep(1.0)


def find_model(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Locate the .eim. Raises with a clear message rather than failing obscurely."""
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"No models directory at {model_dir}. Create it and add your Edge "
            "Impulse .eim (Deployment -> Linux aarch64 / Arduino UNO Q -> Build)."
        )
    candidates = sorted(model_dir.glob("*.eim"))
    if not candidates:
        raise FileNotFoundError(f"No .eim file found in {model_dir}")
    if len(candidates) > 1:
        print(f"[VISION] multiple models found, using {candidates[0].name}")
    return candidates[0]


class Detector:
    """Runs an Edge Impulse .eim directly over its Unix-socket JSON protocol.

    The .eim is a self-contained aarch64 executable. We spawn it, hand it a
    socket path, and exchange newline-delimited JSON. This deliberately avoids
    both the edge_impulse_linux package (not installable here) and the
    object_detection brick (which only accepts models registered via App Lab).
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path or find_model()
        self._proc: object | None = None
        self._sock: socket.socket | None = None
        self._sock_path = ""
        self._next_id = 1
        self._threshold_id: int | None = None
        self.labels: list[str] = []
        self.model_name = "unknown"
        self.input_width = 96
        self.input_height = 96

    def __enter__(self) -> "Detector":
        os.chmod(self._model_path, 0o755)  # the .eim must be executable
        self._sock_path = str(Path(tempfile.gettempdir()) / f"eim-{os.getpid()}.sock")
        if os.path.exists(self._sock_path):
            os.unlink(self._sock_path)

        self._proc = subprocess.Popen(
            [str(self._model_path), self._sock_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        deadline = time.monotonic() + 20.0
        while not os.path.exists(self._sock_path) and time.monotonic() < deadline:
            time.sleep(0.1)
        if not os.path.exists(self._sock_path):
            raise RuntimeError(f"{self._model_path.name} never created its socket")

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(20.0)
        self._sock.connect(self._sock_path)

        hello = self._request({"hello": 1})
        params = hello["model_parameters"]
        self.labels = list(params["labels"])
        self.input_width = int(params["image_input_width"])
        self.input_height = int(params["image_input_height"])
        self.model_name = hello.get("project", {}).get("name", "unknown")
        thresholds = params.get("thresholds") or []
        self._threshold_id = thresholds[0]["id"] if thresholds else None
        print(
            f"[VISION] model '{self.model_name}' "
            f"({self.input_width}x{self.input_height}) labels={self.labels}"
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._sock is not None:
            self._sock.close()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - kill it if it will not go quietly
                self._proc.kill()
        if self._sock_path and os.path.exists(self._sock_path):
            os.unlink(self._sock_path)

    def _request(self, payload: dict) -> dict:
        """Send one JSON message and read its newline-delimited reply."""
        if self._sock is None:
            raise RuntimeError("Detector used outside its context manager")
        payload = dict(payload, id=self._next_id)
        self._next_id += 1

        self._sock.sendall((json.dumps(payload) + "\n").encode())
        buffer = b""
        while b"\n" not in buffer:
            chunk = self._sock.recv(262144)
            if not chunk:
                raise RuntimeError("inference runner closed the connection")
            buffer += chunk
        reply = json.loads(buffer.split(b"\n", 1)[0])
        if not reply.get("success", False):
            raise RuntimeError(f"inference runner error: {reply.get('error')}")
        return reply

    def set_threshold(self, min_score: float, threshold_id: int | None = None) -> None:
        """Lower the runner's own detection threshold.

        The runner filters boxes before returning them, so a client-side
        confidence filter can never see anything below this value.
        """
        if threshold_id is None:
            threshold_id = self._threshold_id
        if threshold_id is None:
            print("[VISION] no configurable threshold on this model")
            return
        self._request({"set_threshold": {"id": threshold_id, "min_score": min_score}})
        print(f"[VISION] runner threshold set to {min_score}")

    def _features(self, frame: Image.Image) -> list[int]:
        """Resize to the model's input and pack each pixel as 0xRRGGBB."""
        small = frame.resize((self.input_width, self.input_height), Image.BILINEAR)
        pixels = np.asarray(small, dtype=np.uint32)
        packed = (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]
        return packed.reshape(-1).tolist()

    def detect(self, frame: Image.Image, min_confidence: float = 0.5) -> list[Detection]:
        """Run one inference. Returns FOMO centroids above min_confidence."""
        reply = self._request({"classify": self._features(frame)})
        boxes = reply.get("result", {}).get("bounding_boxes", [])

        detections: list[Detection] = []
        for item in boxes:
            if item["value"] < min_confidence:
                continue
            # Coordinates are in model input space, not source frame space.
            cx = (item["x"] + item["width"] / 2.0) / self.input_width
            cy = (item["y"] + item["height"] / 2.0) / self.input_height
            detections.append(
                Detection(
                    label=item["label"],
                    confidence=float(item["value"]),
                    x=float(cx),
                    y=float(cy),
                )
            )
        return detections


def best(detections: list[Detection], label: str) -> Detection | None:
    """Highest-confidence detection for one label, or None."""
    matches = [d for d in detections if d.label == label]
    return max(matches, key=lambda d: d.confidence) if matches else None
