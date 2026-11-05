"""The webcam, as numbers: every frame's face (MediaPipe Face Landmarker: 478 landmarks and
the face's pose), the iris found again in the full 1920x1080 frame (`iris`: MediaPipe puts
its landmarks from a ~256 px crop of the face, where a degree of eye turn is a third of a
pixel), and the head as the pointer takes it (`feats`).

The camera serves one program at a time: `run` keeps it and, while sharing is on (`share`,
turned on and off as it goes: `dune-eye-tracker share`), writes every frame to a virtual
camera (v4l2loopback, `SHARE`) that calls and dune-camera read as a webcam meanwhile. Images stay in memory: none is saved, nor sent anywhere else.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import iris
from .face import EYES, model_path, tracker_features

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SIZE = (1920, 1080)  # the frame: the iris needs the pixels
# the virtual camera `run` shares the webcam through (v4l2loopback, `video_nr=10`)
SHARE = "/dev/video10"
# below this the eyes are taken as closed (a blink): the frame says nothing of the gaze
BLINK = 0.12
# the head, as the pointer takes it: where the face points (`head_x`, `head_y`: the x and y
# of its forward axis; looking left turns `head_x` up), where it is (cm), and how open the
# eyes are (lid to lid over the corners' width; low while blinking)
HEAD = ("head_x", "head_y", "pos_x", "pos_y", "pos_z", "open")


def head(landmarks: np.ndarray, matrix: np.ndarray, image) -> dict[str, float]:
    """`HEAD` of one frame (landmarks normalised (478, 3), pose (4, 4))."""
    f = tracker_features(np.asarray(landmarks, float)[None], np.asarray(matrix, float)[None], image)
    return {k: float(f[k][0]) for k in HEAD}


@dataclass
class Frame:
    seq: int
    t: float  # time.monotonic()
    landmarks: np.ndarray | None  # (478, 3) float32, x y normalised to the image; None: no face
    matrix: np.ndarray | None  # (4, 4) float32
    image: tuple[int, int]  # the camera's width, height
    eyes: dict | None = None  # `iris.both`: `limbus`, `dark`, `maps` (and the `crops` when kept)
    feats: dict[str, float] | None = None  # `HEAD`; None: no face


class Camera(threading.Thread):
    """Reads the webcam and runs the landmarker on every frame; `frame()` is the last one.
    `keep_crops`: the eye crops kept in each frame (`record`/`live --keep-eyes`); `share`:
    the virtual camera every frame can go to, `sharing` whether it starts going there
    (`set_sharing` turns it on and off as it runs)."""

    def __init__(self, device: int | str = 0, size: tuple[int, int] = SIZE, share: str | None = None, keep_crops: bool = False, sharing: bool = True):
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
                f"cannot read the camera {device!r}: is it in use (dune-eye-tracker run, dune-camera)? stop it and try again"
                f" (while `run` goes, other programs read the virtual camera {SHARE})"
            )
        self.share, self.keep_crops = share, keep_crops
        self.shared: str | None = None  # what came of sharing: "sharing the camera on ...", or why not
        self.sharing = threading.Event()  # sharing asked for: the camera thread opens or closes the device
        if share and sharing:
            self.sharing.set()
        self.lock = threading.Lock()
        self.latest = Frame(0, 0.0, None, None, size)
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
        except Exception as e:  # shown by whoever reads the frames
            self.error = str(e)
            return
        out = None
        start = time.monotonic()
        seq, frames, since = 0, 0, start
        while not self.stopped.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.error = "the camera stopped giving frames"
                break
            if self.sharing.is_set() and out is None:
                out = self.open_share()
                if out is None:  # it could not: asked again, it tries again
                    self.sharing.clear()
            elif not self.sharing.is_set() and out is not None:
                out = self.close_share(out)
                self.shared = f"not sharing the camera on {self.share} any more"
            if out is not None:
                try:
                    out.send(frame)
                except Exception as e:  # sharing must never stop the tracker
                    self.shared, out = f"stopped sharing the camera: {e}", self.close_share(out)
                    self.sharing.clear()
            now = time.monotonic()
            rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            result = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now - start) * 1000))
            h, w = frame.shape[:2]
            pts = matrix = eyes = feats = None
            if result.face_landmarks and result.facial_transformation_matrixes:
                pts = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
                matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
                feats = head(pts, matrix, (w, h))
                try:  # the iris in the full frame: a failure there must not stop the camera
                    eyes = iris.both(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY), pts, EYES)
                    if not self.keep_crops:
                        eyes.pop("crops")
                except Exception:
                    eyes = None
            seq += 1
            frames += 1
            if now - since >= 1.0:
                self.fps, frames, since = frames / (now - since), 0, now
            with self.lock:
                self.latest = Frame(seq, now, pts, matrix, (w, h), eyes, feats)
        lm.close()  # here, in its own thread: at Python's exit MediaPipe aborts
        self.close_share(out)
        self.cap.release()

    def open_share(self):
        """The virtual camera (`share`), the real one's size and rate; None when there is
        none (`shared` says why: no v4l2loopback device, see the README)."""
        if not self.share:
            return None
        if not Path(self.share).exists():
            self.shared = f"not sharing the camera: no {self.share} (v4l2loopback not loaded, see the README)"
            return None
        try:
            import pyvirtualcam

            w, h = int(self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
            out = pyvirtualcam.Camera(w, h, 30, fmt=pyvirtualcam.PixelFormat.BGR, device=self.share, backend="v4l2loopback")
        except Exception as e:
            self.shared = f"not sharing the camera on {self.share}: {e}"
            return None
        self.shared = f"sharing the camera on {self.share} ({w}x{h})"
        return out

    def set_sharing(self, on: bool) -> bool:
        """Share the camera from the next frame on, or stop; False when there is nothing to
        share it on (no `share`)."""
        if not self.share:
            return False
        (self.sharing.set if on else self.sharing.clear)()
        return True

    @staticmethod
    def close_share(out):
        if out is not None:
            try:
                out.close()
            except Exception:
                pass
        return None

    def frame(self) -> Frame:
        with self.lock:
            return self.latest

    def stop(self):
        self.stopped.set()
