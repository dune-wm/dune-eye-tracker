"""Synthetic eyes: a known head and gaze projected as MediaPipe would give them (the tests'
recordings; the geometry must give them back). Shared with dune-eye-tracker's tests."""

import numpy as np

from . import face
from .recording import BLOCKS, Recording

IMAGE = (1280, 720)
SCREEN = (2560, 1080)
PX_CM = 38.2  # a 67 cm wide picture
BACK, KAPPA = 1.1, (0.05, -0.03)


def rot(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return ry @ rx


def to_plane(px):
    """Screen px to the camera's plane (cm): the camera at the top middle of the screen."""
    return np.array([(px[0] - SCREEN[0] / 2) / PX_CM, -px[1] / PX_CM])


def frame(canon, target_px, head=(0.0, 0.0), pos=(0.0, 3.0, -40.0), back=BACK, kappa=KAPPA, noise=0.0, rng=None):
    """The landmarks (478, 3) and pose (4, 4) of someone at `pos` (cm), the head turned
    `head` (yaw, pitch rad), looking at `target_px`."""
    r = rot(*head)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, pos
    aspect = IMAGE[0] / IMAGE[1]
    pts = canon @ r.T + np.array(pos)
    lm = np.zeros((478, 3))
    lm[:468, :2] = face.project(pts, aspect)
    p = np.array([*to_plane(target_px), 0.0])
    for eye, first in ((face.RIGHT_EYE, 468), (face.LEFT_EYE, 473)):
        centre = r @ face.eyeball(canon, eye, back) + pos
        v_h = r.T @ ((p - centre) / np.linalg.norm(p - centre))
        yaw, pitch = face.angles(v_h[None])
        o = r @ face.direction(yaw - kappa[0], pitch - kappa[1])[0]
        uv = face.project((centre + face.RADIUS * o)[None], aspect)[0]
        if noise and rng is not None:
            uv = uv + rng.normal(0, noise, 2)
        lm[first : first + 5, :2] = uv
    return lm, m


def synthetic(canon, rng, head_sd=0.08, noise=0.0, pos_sd=1.0):
    """A recording: the calibration, head and grid points looked at, the head and its
    place moving about, 12 frames a point."""
    from .record import CALIBRATION, GRID, HEAD

    rows = {k: [] for k in ("landmarks", "matrix", "t", "block", "point", "target", "since")}
    t = 0.0
    for b, points in enumerate((CALIBRATION, HEAD, GRID)):
        for i, (x, y) in enumerate(points):
            target = (x * SCREEN[0], y * SCREEN[1])
            head = rng.normal(0, head_sd, 2)
            pos = (rng.normal(0, pos_sd), 3 + rng.normal(0, pos_sd), -40 + rng.normal(0, pos_sd))
            for k in range(12):
                h = head + (rng.normal(0, 0.06, 2) if BLOCKS[b] == "head" else 0)
                lm, m = frame(canon, target, h, pos, noise=noise, rng=rng)
                since = 1.0 + 0.05 * k
                for key, v in (("landmarks", lm), ("matrix", m), ("t", t), ("block", b), ("point", i), ("target", target), ("since", since)):
                    rows[key].append(v)
                t += 0.033
    a = {k: np.array(v) for k, v in rows.items()}
    return Recording(a["landmarks"], a["matrix"], a["t"], a["block"], a["point"], a["target"], a["since"],
                     {"screen": list(SCREEN), "image": list(IMAGE)}, "synthetic")


def with_iris(rec: Recording) -> Recording:
    """`rec` with the full-frame iris as if found where MediaPipe put it (`dark`, `limbus`)."""
    n = len(rec.t)
    px = rec.landmarks[:, [468, 473], :2] * np.array(rec.image, dtype=float)
    rec.dark = px.copy()
    rec.limbus = np.concatenate([px, np.full((n, 2, 1), 13.0), np.full((n, 2, 1), 20.0)], axis=2)
    rec.maps = np.zeros((n, 2, 2, 3))
    return rec
