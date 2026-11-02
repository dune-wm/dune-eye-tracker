"""What the camera sees, as numbers: the irises in their eyes, and the head.

MediaPipe Face Landmarker gives 478 points of the face (the last 10 the irises) and the
face's pose. From them:

- `eye_u`, `eye_v`: where the iris is in its eye, along the line between the corners
  (`u`, image left to right) and across it (`v`, down), in eye widths from the middle of
  the corners; the two eyes averaged. Moving the eyes moves these.
- `head_x`, `head_y`: where the face points (the x and y of its forward axis).
- `pos_x`, `pos_y`, `pos_z`: where the head is in front of the camera (cm).
- `open`: how open the eyes are (lid to lid over the width); low while blinking.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Landmark indices (MediaPipe face mesh): corners and lids of each eye, image left to
# right, and the iris centre.
RIGHT_EYE = {"left": 33, "right": 133, "top": 159, "bottom": 145, "iris": 468}
LEFT_EYE = {"left": 362, "right": 263, "top": 386, "bottom": 374, "iris": 473}

NAMES = ("eye_u", "eye_v", "head_x", "head_y", "pos_x", "pos_y", "pos_z", "open")

# Below this the eyes are taken as closed (a blink): the sample says nothing of the gaze.
BLINK = 0.12

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def model_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "dune-eye-tracker" / "face_landmarker.task"


def eye(points: np.ndarray, idx: dict[str, int]) -> tuple[float, float, float]:
    """(u, v, openness) of one eye; `points` in pixels, shape (478, 2)."""
    left, right = points[idx["left"]], points[idx["right"]]
    width = float(np.linalg.norm(right - left))
    if width < 1e-6:
        return 0.0, 0.0, 0.0
    axis = (right - left) / width
    across = np.array([-axis[1], axis[0]])  # the axis turned a quarter: down in the image
    rel = points[idx["iris"]] - (left + right) / 2
    openness = float(np.linalg.norm(points[idx["top"]] - points[idx["bottom"]])) / width
    return float(rel @ axis) / width, float(rel @ across) / width, openness


def features(points: np.ndarray, matrix: np.ndarray | None) -> dict[str, float]:
    """The numbers of one frame: `points` in pixels (478, 2), `matrix` the face pose (4x4)."""
    ru, rv, ro = eye(points, RIGHT_EYE)
    lu, lv, lo = eye(points, LEFT_EYE)
    out = {"eye_u": (ru + lu) / 2, "eye_v": (rv + lv) / 2, "open": (ro + lo) / 2}
    if matrix is not None:
        out |= {
            "head_x": float(matrix[0, 2]),
            "head_y": float(matrix[1, 2]),
            "pos_x": float(matrix[0, 3]),
            "pos_y": float(matrix[1, 3]),
            "pos_z": float(matrix[2, 3]),
        }
    else:
        out |= dict.fromkeys(("head_x", "head_y", "pos_x", "pos_y", "pos_z"), 0.0)
    return out


@dataclass
class Sample:
    seq: int
    t: float
    feats: dict[str, float] | None  # None: no face


class Camera(threading.Thread):
    """Reads the webcam and runs the landmarker on every frame; `latest` is the last one."""

    def __init__(self, device: int | str = 0, size: tuple[int, int] = (1280, 720)):
        super().__init__(daemon=True)
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened() or not self.cap.read()[0]:
            raise RuntimeError(
                f"cannot read the camera {device!r}: is it in use (dune-camera)? close it and try again"
            )
        self.lock = threading.Lock()
        self.latest = Sample(0, 0.0, None)
        self.fps = 0.0
        self.stopped = threading.Event()
        self.error: str | None = None

    def landmarker(self):
        import mediapipe as mp  # noqa: F401  (loads the native library)
        from mediapipe.tasks.python import BaseOptions, vision

        path = model_path()
        if not path.exists():
            raise RuntimeError(f"no model at {path}: download it with\n  curl -L --create-dirs -o {path} {MODEL_URL}")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def run(self):
        import mediapipe as mp

        try:
            lm = self.landmarker()
        except Exception as e:  # shown by the window
            self.error = str(e)
            return
        start = time.monotonic()
        seq, frames, since = 0, 0, start
        while not self.stopped.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.error = "the camera stopped giving frames"
                return
            now = time.monotonic()
            rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            result = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now - start) * 1000))
            feats = None
            if result.face_landmarks:
                h, w = frame.shape[:2]
                pts = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
                matrix = np.array(result.facial_transformation_matrixes[0]) if result.facial_transformation_matrixes else None
                feats = features(pts, matrix)
            seq += 1
            frames += 1
            if now - since >= 1.0:
                self.fps, frames, since = frames / (now - since), 0, now
            with self.lock:
                self.latest = Sample(seq, now, feats)
        lm.close()  # here, in its own thread: at Python's exit MediaPipe aborts
        self.cap.release()

    def sample(self) -> Sample:
        with self.lock:
            return self.latest

    def stop(self):
        self.stopped.set()
