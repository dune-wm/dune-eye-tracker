"""Ways from a frame's landmarks and pose to a point on the screen, all fitted the same way:
on some frames of a recording whose targets are known (`fit`), then guessing others
(`predict`). `MODELS` is what `eval` compares; add one there to try an idea on every
recording made.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from functools import partial

from . import face, iris
from .recording import Recording

SOURCES = ("mediapipe", "limbus", "dark")  # where the iris centre comes from

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


# --- ridge on a polynomial (dune-eye-tracker's model.fit) ---------------------------


def design(x: np.ndarray, quad: list[int]) -> np.ndarray:
    """1, the columns, and the products of each pair (with itself) of `quad`."""
    cols = [np.ones(len(x)), *x.T]
    for i, a in enumerate(quad):
        for b in quad[i:]:
            cols.append(x[:, a] * x[:, b])
    return np.column_stack(cols)


def ridge(a: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    reg = lam * np.eye(a.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(a.T @ a + reg, a.T @ y)


class Ridge:
    """A quadratic ridge on named per-frame features, the strength chosen by leaving out
    one target at a time. `features(rec, idx)` -> {name: (n,)}."""

    def __init__(self, name: str, columns: tuple[str, ...], quad: tuple[str, ...], features):
        self.name, self.columns, self.quad, self.features = name, columns, quad, features

    def _x(self, rec: Recording, idx: np.ndarray) -> np.ndarray:
        f = self.features(rec, idx)
        return np.column_stack([f[c] for c in self.columns])

    def fit(self, rec: Recording, idx: np.ndarray, groups: np.ndarray):
        x = self._x(rec, idx)
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        a = design((x - self.mean) / self.scale, [self.columns.index(c) for c in self.quad])
        y = rec.target[idx]
        best = None
        for lam in LAMBDAS:
            err = []
            for g in np.unique(groups):
                out = groups == g
                w = ridge(a[~out], y[~out], lam)
                err.append(np.linalg.norm(a[out] @ w - y[out], axis=1).mean())
            if best is None or np.mean(err) < best[0]:
                best = (float(np.mean(err)), lam)
        self.lam = best[1]
        self.w = ridge(a, y, self.lam)
        return self

    def predict(self, rec: Recording, idx: np.ndarray) -> np.ndarray:
        x = (self._x(rec, idx) - self.mean) / self.scale
        return design(x, [self.columns.index(c) for c in self.quad]) @ self.w


_found: dict = {}


def found(rec: Recording) -> tuple[np.ndarray, np.ndarray] | None:
    """`limbus` (n, 2, 4) and `dark` (n, 2, 2) of a recording: found again on its eye
    crops when it kept them (the iris finder as it is now), else as recorded."""
    if rec.limbus is None:
        return None
    if rec.crops is None:
        return rec.limbus, rec.dark  # as recorded (live: one frame at a time, never cached)
    key = id(rec)
    if key not in _found:  # the recording is kept alive by `eval` meanwhile: its id stays its own
        w, h = rec.image
        lb, dk = np.full(rec.limbus.shape, np.nan), np.full(rec.dark.shape, np.nan)
        for n in range(len(rec.t)):
            px = rec.landmarks[n, :, :2] * np.array([w, h])
            for k, eye in enumerate(face.EYES):
                lb[n, k], dk[n, k] = iris.locate(rec.crops[n, k], rec.maps[n, k], px[eye["iris"]], iris.iris_radius(px, eye["iris"]))
        _found[key] = (lb, dk)
    return _found[key]


def landmarks(rec: Recording, idx: np.ndarray, source: str = "mediapipe") -> np.ndarray:
    """The landmarks of frames `idx`, the iris centres from `source`: MediaPipe's, or found
    in the full frame (`limbus`, `dark`; MediaPipe's where it was not found)."""
    lm = rec.landmarks[idx]
    f = found(rec) if source != "mediapipe" else None
    if f is None:
        return lm
    pts = (f[0] if source == "limbus" else f[1])[idx, :, :2]
    lm = lm.copy()
    norm = pts / np.array(rec.image, dtype=float)
    for k, eye in enumerate(face.EYES):
        ok = np.all(np.isfinite(norm[:, k]), axis=1)
        lm[ok, eye["iris"], :2] = norm[ok, k]
    return lm


def tracker(rec: Recording, idx: np.ndarray, source: str = "mediapipe") -> dict:
    return face.tracker_features(landmarks(rec, idx, source), rec.matrix[idx], rec.image)


def eyes_in_head(rec: Recording, idx: np.ndarray, source: str = "mediapipe") -> dict:
    """The eyes turning in the head (`yaw`, `pitch`) and the head (as the tracker has it)."""
    a = face.eye_angles(landmarks(rec, idx, source), rec.matrix[idx], rec.aspect, face.canonical())
    return tracker(rec, idx, source) | {"yaw": a[:, 0], "pitch": a[:, 1]}


# --- the geometric model -------------------------------------------------------------


def poly(xy: np.ndarray, order: int) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    cols = [np.ones(len(xy)), x, y]
    if order == 2:
        cols += [x * x, x * y, y * y]
    return np.column_stack(cols)


class Geometric:
    """The visual axes met with the camera's plane (`face.on_plane`), then to pixels by an
    affine map (`order` 1: the monitor's place, size and tilt against the camera) or a
    quadratic one (2). Fitted: the eyeball's depth (`back`), `kappa`, with `gain` the eyes'
    turn per axis too, and the map (solved by least squares at each step), robustly (soft
    L1, 100 px)."""

    def __init__(self, name: str, order: int = 1, gain: bool = False, source: str = "mediapipe", eyes=(0, 1)):
        self.name, self.order, self.gain, self.source, self.eyes = name, order, gain, source, eyes
        lo, hi = [0.2, -0.35, -0.35], [2.5, 0.35, 0.35]
        self.x0, self.scale = [face.BACK, 0.0, 0.0], [0.5, 0.05, 0.05]
        if gain:
            lo, hi = lo + [0.3, 0.3], hi + [4.0, 6.0]
            self.x0, self.scale = self.x0 + [1.0, 1.0], self.scale + [0.3, 0.3]
        self.BOUNDS = (lo, hi)

    def plane(self, rec: Recording, idx: np.ndarray, p=None) -> np.ndarray:
        p = self.p if p is None else p
        gain = (p[3], p[4]) if self.gain else (1.0, 1.0)
        return face.on_plane(landmarks(rec, idx, self.source), rec.matrix[idx], rec.aspect, face.canonical(), p[0], (p[1], p[2]), gain=gain, eyes=self.eyes)

    def fit(self, rec: Recording, idx: np.ndarray, groups: np.ndarray, weights: np.ndarray | None = None):
        """`weights` (per frame, default 1): how much each counts (dune-eye-tracker's clicks
        by what made them)."""
        y = rec.target[idx]
        sw = np.ones(len(idx)) if weights is None else np.sqrt(np.asarray(weights, dtype=float))

        def solve(p):
            d = poly(self.plane(rec, idx, p), self.order)
            w, *_ = np.linalg.lstsq(d * sw[:, None], y * sw[:, None], rcond=None)
            return d, w

        def residual(p):
            d, w = solve(p)
            return ((d @ w - y) * sw[:, None]).ravel()

        res = least_squares(residual, x0=self.x0, bounds=self.BOUNDS, loss="soft_l1", f_scale=100.0, x_scale=self.scale)
        self.p = res.x
        _, self.w = solve(self.p)
        return self

    def predict(self, rec: Recording, idx: np.ndarray) -> np.ndarray:
        return poly(self.plane(rec, idx), self.order) @ self.w


class Binocular:
    """Each eye's own geometric model, the two guesses blended by where on the screen
    they fall: an eye turned in towards the nose is hidden by its corner (looking right,
    the right eye; on 2026-11-04 the right edge was 698 px off with it alone, 247 with the
    left). How much each side leans to the other eye (`KS`) is chosen on the calibration
    points, not set: the two sides need not be alike (light, the camera's place)."""

    KS = (0.0, 0.5, 1.0, 1.5, 2.0)

    def __init__(self, name: str, source: str = "dark", order: int = 1):
        self.name, self.source = name, source
        self.eye = [Geometric(f"{name} right", order, source=source, eyes=(0,)), Geometric(f"{name} left", order, source=source, eyes=(1,))]
        self.k = (0.0, 0.0)

    def _blend(self, pr: np.ndarray, pl: np.ndarray, width: float, k) -> np.ndarray:
        x = (pr[:, 0] + pl[:, 0]) / 2 / width
        wl = np.where(x > 0.5, 0.5 + k[1] * (x - 0.5), 0.5 - k[0] * (0.5 - x))  # how much the left eye counts
        wl = np.clip(wl, 0.0, 1.0)[:, None]
        return wl * pl + (1 - wl) * pr

    def fit(self, rec: Recording, idx: np.ndarray, groups: np.ndarray):
        for m in self.eye:
            m.fit(rec, idx, groups)
        pr, pl = (m.predict(rec, idx) for m in self.eye)
        y = rec.target[idx]

        def err(k):
            g = self._blend(pr, pl, rec.screen[0], k)
            return np.mean([np.linalg.norm(np.median(g[groups == u], axis=0) - y[groups == u][0]) for u in np.unique(groups)])

        self.k = min(((a, b) for a in self.KS for b in self.KS), key=err)
        return self

    def predict(self, rec: Recording, idx: np.ndarray) -> np.ndarray:
        pr, pl = (m.predict(rec, idx) for m in self.eye)
        return self._blend(pr, pl, rec.screen[0], self.k)


def models(with_iris: bool = False) -> list:
    """What `eval` compares: dune-eye-tracker's eyes model and the geometric one, with the
    iris from MediaPipe and (`with_iris`: the recording has it) found in the full frame;
    the tracker's with the head, the geometric quadratic, and a ridge on the eyes turning
    in the head plus the head."""
    head = ("head_x", "head_y", "pos_x", "pos_y", "pos_z")
    out = []
    for src in SOURCES if with_iris else SOURCES[:1]:
        tag = "" if src == "mediapipe" else f" [{src}]"
        out += [
            Ridge("tracker: eyes" + tag, ("eye_u", "eye_v"), ("eye_u", "eye_v"), partial(tracker, source=src)),
            Geometric("geometric" + tag, 1, source=src),
        ]
    if with_iris:
        out.append(Binocular("geometric, eyes weighted [dark]", "dark"))
    src = "limbus" if with_iris else "mediapipe"
    tag = "" if src == "mediapipe" else f" [{src}]"
    out += [
        Ridge("tracker: eyes + head", ("eye_u", "eye_v", "head_x", "head_y", "pos_x", "pos_y"), ("eye_u", "eye_v"), tracker),
        Geometric("geometric, quadratic" + tag, 2, source=src),
        Ridge("eyes in head + head" + tag, ("yaw", "pitch", *head), ("yaw", "pitch"), partial(eyes_in_head, source=src)),
    ]
    return out
