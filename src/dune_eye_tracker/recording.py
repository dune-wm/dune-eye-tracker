"""A recording: every frame's landmarks and pose while known points were looked at.

Saved as one `.npz` (`save`): `landmarks` (n, 478, 3) float32, `matrix` (n, 4, 4) float32,
`t` (n,) s, `block` (n,) (`BLOCKS`), `point` (n,) the point's number in its block, `target`
(n, 2) screen px, `since` (n,) s since that point showed, `meta` (JSON: screen and image
size, time, how long a point showed per block); and the iris found in the full frame
(`iris`): `limbus` (n, 2, 4) x y radius inliers, `dark` (n, 2, 2), `maps` (n, 2, 2, 3) the eye
crops' maps, and the crops themselves (`crops`, (n, 2, h, w) uint8) only when asked. Only
frames with a face are kept.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# block numbers 0..3; `click`: dune-eye-tracker's clicks in use, the point where you
# clicked (you look at the pointer), its frames from just before it (`since` < 0)
BLOCKS = ("calibration", "head", "grid", "click")

# s after a point shows: the frames taken as on it (the eyes there, steady), per block; a
# click's are before it (the eyes there already, the hand not yet: as dune-eye-tracker's)
WINDOWS = {"calibration": (0.6, 1.5), "head": (0.8, 4.3), "grid": (0.6, 1.5), "click": (-0.35, -0.05)}


def state_dir() -> Path:
    """Where dune-eye-tracker keeps its numbers (calibration, clicks, recordings, games)."""
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "dune-eye-tracker"


def recordings_dir() -> Path:
    return state_dir() / "recordings"


def lab_recordings_dir() -> Path:
    """dune-gaze's, where the gaze was worked out (2026-11-03..04): read, not written."""
    return state_dir().parent / "dune-gaze" / "recordings"


@dataclass
class Recording:
    landmarks: np.ndarray
    matrix: np.ndarray
    t: np.ndarray
    block: np.ndarray
    point: np.ndarray
    target: np.ndarray
    since: np.ndarray
    meta: dict = field(default_factory=dict)
    name: str = ""
    limbus: np.ndarray | None = None
    dark: np.ndarray | None = None
    maps: np.ndarray | None = None
    crops: np.ndarray | None = None

    @property
    def image(self) -> tuple[int, int]:
        return tuple(self.meta["image"])

    @property
    def screen(self) -> tuple[int, int]:
        return tuple(self.meta["screen"])

    @property
    def aspect(self) -> float:
        w, h = self.image
        return w / h

    def select(self, block: str, window: tuple[float, float] | None = None) -> np.ndarray:
        """The frames of `block` within its window (indices)."""
        lo, hi = window or WINDOWS[block]
        return np.flatnonzero((self.block == BLOCKS.index(block)) & (self.since >= lo) & (self.since <= hi))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {k: getattr(self, k) for k in ("limbus", "dark", "maps", "crops") if getattr(self, k) is not None}
        np.savez_compressed(
            path, landmarks=self.landmarks.astype(np.float32), matrix=self.matrix.astype(np.float32), t=self.t,
            block=self.block.astype(np.int8), point=self.point.astype(np.int16), target=self.target.astype(np.float32),
            since=self.since.astype(np.float32), meta=np.array(json.dumps(self.meta)), **extra,
        )

    @classmethod
    def load(cls, path: Path) -> "Recording":
        with np.load(path) as z:
            extra = {k: (z[k] if k == "crops" else z[k].astype(float)) for k in ("limbus", "dark", "maps", "crops") if k in z}
            return cls(
                z["landmarks"].astype(float), z["matrix"].astype(float), z["t"], z["block"].astype(int), z["point"].astype(int),
                z["target"].astype(float), z["since"].astype(float), json.loads(str(z["meta"])), Path(path).stem, **extra,
            )


def build(frames: list[dict], meta: dict) -> Recording:
    """A recording from the frames kept while recording (each `landmarks`, `matrix`, `t`,
    `block`, `point`, `target`, `since`)."""
    def col(k, dtype=float):
        return np.array([f[k] for f in frames], dtype=dtype)

    rec = Recording(col("landmarks"), col("matrix"), col("t"), col("block", int), col("point", int), col("target"), col("since"), meta)
    if all(f.get("limbus") is not None for f in frames):
        rec.limbus, rec.dark, rec.maps = col("limbus"), col("dark"), col("maps")
        if all(f.get("crops") is not None for f in frames):
            rec.crops = col("crops", np.uint8)
    return rec


def all_recordings() -> list[Path]:
    """Every recording, dune-gaze's too, oldest first (by name: the time it was made)."""
    paths = [p for d in (lab_recordings_dir(), recordings_dir()) if d.exists() for p in d.glob("*.npz")]
    return sorted(paths, key=lambda p: p.name)
