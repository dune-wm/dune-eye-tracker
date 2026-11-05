"""The face in 3D: MediaPipe's canonical face, its camera, and the eyes in the head.

MediaPipe Face Landmarker gives, for each frame, 478 landmarks (x, y normalised to the
image, the last 10 the irises) and a pose: the 4x4 matrix that puts its canonical face
(468 points, cm, x to the face's left... as seen: x right, y up, z out of the face) in the
camera's space (cm, the camera at 0 looking down -z, y up), seen through a perspective
camera of vertical field `VFOV`. With the two:

- the eyeballs are where the canonical face has them (behind the middle of each eye's
  corners, `back` cm), moved by the pose: they turn with the head;
- the iris is on the eyeball's sphere (`RADIUS`) where the camera's ray through its image
  point meets it; eyeball centre to iris is the eye's optical axis;
- that axis in the head's own frame (`yaw`, `pitch`) is the eye turning in the head,
  whatever the head does; turned back into the camera's frame (plus `kappa`, the visual
  axis's offset) it is a ray from the eye, met with the camera's plane (z = 0: a webcam on
  the monitor's edge) at `X`, `Y` cm. The two eyes' points averaged.
"""

from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

VFOV = 63.0  # degrees: MediaPipe's face geometry camera
RADIUS = 1.2  # cm: an eyeball's radius (the iris on its sphere)
BACK = 1.0  # cm: how far behind the middle of the eye's corners its centre is (fitted)

# landmarks: corners (image left, right), lids, iris centre of each eye
RIGHT_EYE = {"left": 33, "right": 133, "top": 159, "bottom": 145, "iris": 468}
LEFT_EYE = {"left": 362, "right": 263, "top": 386, "bottom": 374, "iris": 473}
EYES = (RIGHT_EYE, LEFT_EYE)


def model_path() -> Path:
    """The face landmarker, shared with dune-eye-tracker."""
    import os

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "dune-eye-tracker" / "face_landmarker.task"


# --- the canonical face -----------------------------------------------------------


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b: bytes):
    """(field, wire type, value) of a protobuf message."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i : i + n], i + n
        elif wt == 5:
            v, i = b[i : i + 4], i + 4
        elif wt == 1:
            v, i = b[i : i + 8], i + 8
        else:
            raise ValueError(f"wire type {wt}")
        yield f, wt, v


@lru_cache(maxsize=1)
def canonical(path: str | None = None) -> np.ndarray:
    """The canonical face's 468 points (cm), from the landmarker's bundle: its geometry
    metadata's mesh (field 1), whose vertex buffer (field 3) is x, y, z, u, v a point."""
    with zipfile.ZipFile(path or model_path()) as z:
        meta = z.read("geometry_pipeline_metadata_landmarks.binarypb")
    mesh = next(v for f, _, v in _fields(meta) if f == 1)
    vb = [(wt, v) for f, wt, v in _fields(mesh) if f == 3]
    if len(vb) == 1 and vb[0][0] == 2:  # packed
        floats = np.frombuffer(vb[0][1], dtype="<f4")
    else:
        floats = np.frombuffer(b"".join(v for _, v in vb), dtype="<f4")
    return floats.reshape(-1, 5)[:, :3].astype(float)


# --- the camera -------------------------------------------------------------------


def project(points: np.ndarray, aspect: float, vfov: float = VFOV) -> np.ndarray:
    """Camera-space points (..., 3) to normalised image points (..., 2): x right, y down."""
    t = np.tan(np.radians(vfov) / 2)
    z = -points[..., 2]
    xn = points[..., 0] / z / (t * aspect)
    yn = points[..., 1] / z / t
    return np.stack([(xn + 1) / 2, (1 - yn) / 2], axis=-1)


def ray(uv: np.ndarray, aspect: float, vfov: float = VFOV) -> np.ndarray:
    """Normalised image points (..., 2) to the unit rays from the camera through them."""
    t = np.tan(np.radians(vfov) / 2)
    d = np.stack([(2 * uv[..., 0] - 1) * t * aspect, (1 - 2 * uv[..., 1]) * t, -np.ones(uv.shape[:-1])], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(linear part, its rotation, its scale) of pose matrices (n, 4, 4)."""
    lin = matrix[:, :3, :3]
    scale = np.cbrt(np.abs(np.linalg.det(lin)))
    return lin, lin / scale[:, None, None], scale


def to_camera(matrix: np.ndarray, p: np.ndarray) -> np.ndarray:
    """A head-frame point (3,) or points (n, 3) in camera space, per frame (n, 4, 4)."""
    p = np.broadcast_to(p, (len(matrix), 3))
    return np.einsum("nij,nj->ni", matrix[:, :3, :3], p) + matrix[:, :3, 3]


# --- the eyes in the head ---------------------------------------------------------


def eyeball(canon: np.ndarray, eye: dict, back: float) -> np.ndarray:
    """The centre of an eyeball in the head's frame (cm)."""
    mid = (canon[eye["left"]] + canon[eye["right"]]) / 2
    return mid - np.array([0.0, 0.0, back])


def optical(lm: np.ndarray, matrix: np.ndarray, aspect: float, canon: np.ndarray, back: float = BACK, radius: float = RADIUS, vfov: float = VFOV):
    """Per eye, per frame: the eyeball's centre in camera space (n, 3) and its optical
    axis in the head's frame (n, 3). `lm` (n, 478, >=2) normalised landmarks."""
    _, rot, scale = pose(matrix)
    out = []
    for eye in EYES:
        centre = to_camera(matrix, eyeball(canon, eye, back))
        d = ray(lm[:, eye["iris"], :2], aspect, vfov)
        r = radius * scale
        b = np.einsum("ni,ni->n", d, centre)
        disc = b**2 - np.einsum("ni,ni->n", centre, centre) + r**2
        s = np.where(disc >= 0, b - np.sqrt(np.maximum(disc, 0.0)), b)  # the near side; missed: the closest point
        iris = d * s[:, None]
        axis = iris - centre
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        out.append((centre, np.einsum("nji,nj->ni", rot, axis)))  # rot^T: into the head's frame
    return out


def angles(axis_head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(yaw, pitch) radians of head-frame directions (n, 3): + to the head's x, + up."""
    return np.arctan2(axis_head[:, 0], axis_head[:, 2]), np.arcsin(np.clip(axis_head[:, 1], -1, 1))


def direction(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], axis=-1)


def on_plane(lm: np.ndarray, matrix: np.ndarray, aspect: float, canon: np.ndarray, back: float = BACK, kappa=(0.0, 0.0), radius: float = RADIUS, vfov: float = VFOV, gain=(1.0, 1.0), eyes=(0, 1)) -> np.ndarray:
    """Where the eyes' visual axes meet the camera's plane (z = 0), cm (n, 2): each eye's
    turn in the head (yaw, pitch) times `gain` (the iris landmarks move less than the eye
    turns, up and down most: the lids), plus `kappa` (radians), the `eyes` (0 the right
    eye, image left; 1 the left) averaged."""
    _, rot, _ = pose(matrix)
    pts = []
    for k, (centre, axis) in enumerate(optical(lm, matrix, aspect, canon, back, radius, vfov)):
        if k not in eyes:
            continue
        yaw, pitch = angles(axis)
        v = np.einsum("nij,nj->ni", rot, direction(gain[0] * yaw + kappa[0], gain[1] * pitch + kappa[1]))
        vz = np.where(np.abs(v[:, 2]) < 1e-6, 1e-6, v[:, 2])
        s = -centre[:, 2] / vz
        pts.append(centre[:, :2] + v[:, :2] * s[:, None])
    return sum(pts) / len(pts)


def eye_angles(lm: np.ndarray, matrix: np.ndarray, aspect: float, canon: np.ndarray, back: float = BACK) -> np.ndarray:
    """The eyes turning in the head (yaw, pitch radians, the two averaged) (n, 2)."""
    a = [np.column_stack(angles(axis)) for _, axis in optical(lm, matrix, aspect, canon, back)]
    return (a[0] + a[1]) / 2


# --- the 2D features of dune-eye-tracker ------------------------------------------


def tracker_features(lm: np.ndarray, matrix: np.ndarray, image: tuple[int, int]) -> dict[str, np.ndarray]:
    """dune-eye-tracker's `features.features`, for every frame: `eye_u`, `eye_v` (the iris
    along / across its eye's corners, in eye widths, image pixels), `open`, `head_x`,
    `head_y`, `pos_x`, `pos_y`, `pos_z`."""
    px = lm[..., :2] * np.array(image, dtype=float)
    us, vs, opens = [], [], []
    for eye in EYES:
        left, right = px[:, eye["left"]], px[:, eye["right"]]
        width = np.linalg.norm(right - left, axis=1)
        width = np.where(width < 1e-6, 1e-6, width)
        axis = (right - left) / width[:, None]
        across = np.stack([-axis[:, 1], axis[:, 0]], axis=1)
        rel = px[:, eye["iris"]] - (left + right) / 2
        us.append(np.einsum("ni,ni->n", rel, axis) / width)
        vs.append(np.einsum("ni,ni->n", rel, across) / width)
        opens.append(np.linalg.norm(px[:, eye["top"]] - px[:, eye["bottom"]], axis=1) / width)
    return {
        "eye_u": (us[0] + us[1]) / 2, "eye_v": (vs[0] + vs[1]) / 2, "open": (opens[0] + opens[1]) / 2,
        "head_x": matrix[:, 0, 2], "head_y": matrix[:, 1, 2],
        "pos_x": matrix[:, 0, 3], "pos_y": matrix[:, 1, 3], "pos_z": matrix[:, 2, 3],
    }


def reprojection(lm: np.ndarray, matrix: np.ndarray, image: tuple[int, int], canon: np.ndarray, vfov: float = VFOV) -> np.ndarray:
    """Per frame, the median distance (image px) between the landmarks and the canonical
    face put there by the pose: whether the pose and the camera model hold."""
    aspect = image[0] / image[1]
    pts = np.einsum("nij,kj->nki", matrix[:, :3, :3], canon) + matrix[:, None, :3, 3]
    uv = project(pts, aspect, vfov)
    d = (uv - lm[:, : len(canon), :2]) * np.array(image, dtype=float)
    return np.median(np.linalg.norm(d, axis=-1), axis=1)
