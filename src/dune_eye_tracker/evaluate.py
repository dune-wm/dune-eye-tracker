"""How far off each model is: fitted on a recording's calibration points, tested on its
grid (the head as it sits) and on its head-moving points (the eyes on a point, the head
turning: a head-invariant gaze stays on it), and across recordings (fitted on one day's
calibration, tested on another's grid: another posture). Medians with a 95% interval.
"""

from __future__ import annotations

import numpy as np

from . import face
from .models import models
from .recording import BLOCKS, Recording

BLINK = 0.12  # eye openness under which a frame is a blink
MIN_FRAMES = 3
QUADRANTS = ("top left", "top right", "bottom left", "bottom right")
GOAL = (100.0, 120.0)


def ci(xs, n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 3:
        return None
    meds = np.median(np.random.default_rng(seed).choice(xs, size=(n, len(xs))), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def med_ci(xs) -> str:
    if not len(xs):
        return "-"
    c = ci(xs)
    return f"{np.median(xs):.0f}" + ("" if c is None else f" ({c[0]:.0f}-{c[1]:.0f})")


def quadrant(x: float, y: float, screen) -> int:
    return (2 if y >= screen[1] / 2 else 0) + (1 if x >= screen[0] / 2 else 0)


def frames(rec: Recording, block: str) -> np.ndarray:
    """The frames of `block` in its window, eyes open."""
    idx = rec.select(block)
    if not len(idx):
        return idx
    opened = face.tracker_features(rec.landmarks[idx], rec.matrix[idx], rec.image)["open"] >= BLINK
    return idx[opened]


def on_screen(guess: np.ndarray, screen) -> np.ndarray:
    """Guesses kept on the screen: you look at it (off it a guess is only wrong; on 3
    recordings this took 189 px to 164)."""
    return np.clip(guess, 0, np.array(screen, dtype=float) - 1)


def groups(rec: Recording, idx: np.ndarray) -> np.ndarray:
    return rec.block[idx] * 1000 + rec.point[idx]


def misses(model, rec: Recording, block: str, per_frame: bool = False) -> list[tuple[int, float]]:
    """(quadrant, px) per point of `block`: the median guess from the target, or (with
    `per_frame`, for the head-moving points, where the guesses swing both ways around it)
    the median distance of each frame's guess."""
    idx = frames(rec, block)
    if not len(idx):
        return []
    guess = on_screen(model.predict(rec, idx), rec.screen)
    out = []
    for g in np.unique(groups(rec, idx)):
        m = groups(rec, idx) == g
        if m.sum() < MIN_FRAMES:
            continue
        target = rec.target[idx][m][0]
        if per_frame:
            d = float(np.median(np.linalg.norm(guess[m] - target, axis=1)))
        else:
            d = float(np.linalg.norm(np.median(guess[m], axis=0) - target))
        out.append((quadrant(*target, rec.screen), d))
    return out


def fitted(rec: Recording, train: tuple[str, ...] = ("calibration",), with_iris: bool | None = None) -> list:
    idx = np.concatenate([frames(rec, b) for b in train])
    if len(np.unique(groups(rec, idx))) < 4:
        return []  # too few points to fit on
    return [m.fit(rec, idx, groups(rec, idx)) for m in models(rec.limbus is not None if with_iris is None else with_iris)]


def iris_found(rec: Recording) -> str:
    """How often the full-frame iris was found, and how it jitters while fixating."""
    from .models import found

    f = found(rec)
    if f is None:
        return "no full-frame iris in this recording (recorded before it, or at 1280x720)"
    lb = f[0]
    ok = np.isfinite(lb[:, :, 0])
    if not ok.any():
        return "iris never found in the full frame"
    g = frames(rec, "grid")
    jit = {}
    for name, pts in (("mediapipe", np.stack([rec.landmarks[:, e["iris"], :2] * np.array(rec.image) for e in face.EYES], 1)), ("limbus", lb[:, :, :2]), ("dark", f[1])):
        sd = []
        for p in np.unique(rec.point[g]):
            m = g[rec.point[g] == p]
            v = pts[m].reshape(len(m), -1)
            v = v[np.all(np.isfinite(v), axis=1)]
            if len(v) >= MIN_FRAMES:
                sd.append(v.std(axis=0).mean())
        jit[name] = np.median(sd) if sd else float("nan")
    return (f"iris found in the full frame: limbus {100 * ok.mean():.0f}% of eyes (radius {np.nanmedian(lb[:, :, 2]):.1f} px); "
            f"jitter while fixating (px): mediapipe {jit['mediapipe']:.2f}, limbus {jit['limbus']:.2f}, dark {jit['dark']:.2f}")


def table(rows: list[tuple[str, list, list]]) -> list[str]:
    """Rows of (model, grid misses, head-moving misses)."""
    out = [f"{'model':30} {'grid (95%)':>15} {'head moving':>15} " + " ".join(f"{q:>12}" for q in QUADRANTS)]
    for name, grid, head in rows:
        per = [[d for q, d in grid if q == k] for k in range(4)]
        out.append(
            f"{name:30} {med_ci([d for _, d in grid]):>15} {med_ci([d for _, d in head]):>15} "
            + " ".join(f"{'-' if not p else f'{np.median(p):.0f}':>12}" for p in per)
        )
    return out


def report(recs: list[Recording], cross: bool = True) -> list[str]:
    out = []
    for rec in recs:
        n = {b: len(np.unique(rec.point[rec.block == i])) for i, b in enumerate(BLOCKS)}
        out += [f"{rec.name}: {n['calibration']} calibration, {n['head']} head-moving, {n['grid']} grid points; screen {rec.screen[0]}x{rec.screen[1]}, camera {rec.image[0]}x{rec.image[1]}", iris_found(rec), ""]
        out.append("fitted on the calibration points:")
        out += table([(m.name, misses(m, rec, "grid"), misses(m, rec, "head", True)) for m in fitted(rec)])
        if n["head"]:
            out += ["", "fitted on the calibration and head-moving points (head moving: in-sample):"]
            out += table([(m.name, misses(m, rec, "grid"), misses(m, rec, "head", True)) for m in fitted(rec, ("calibration", "head"))])
        out.append("")
    if cross and len(recs) > 1:
        out.append("across recordings (fitted on one's calibration, tested on another's grid; same camera size):")
        rows: dict[str, tuple[list, list]] = {}
        for a in recs:
            ms = fitted(a)
            for b in recs:
                # the same screen and camera (and both with the full-frame iris, or neither)
                if b is a or b.screen != a.screen or b.image != a.image or (b.limbus is None) != (a.limbus is None):
                    continue
                for m in ms:
                    r = rows.setdefault(m.name, ([], []))
                    r[0].extend(misses(m, b, "grid"))
                    r[1].extend(misses(m, b, "head", True))
        out += table([(k, g, h) for k, (g, h) in rows.items()])
    out.append(f"\ngoal: {GOAL[0]:.0f}-{GOAL[1]:.0f} px median on the grid")
    return out


def check(rec: Recording) -> list[str]:
    """Whether the pose and the camera model put the canonical face on the landmarks."""
    canon = face.canonical()
    sample = np.linspace(0, len(rec.t) - 1, min(200, len(rec.t))).astype(int)
    lm, mx = rec.landmarks[sample], rec.matrix[sample]
    errs = {v: float(np.median(face.reprojection(lm, mx, rec.image, canon, v))) for v in range(40, 91)}
    best = min(errs, key=errs.get)
    return [
        f"{rec.name}: {len(rec.t)} frames, image {rec.image[0]}x{rec.image[1]}",
        f"the canonical face put by the pose: median {errs[int(face.VFOV)]:.1f} px off the landmarks at {face.VFOV:g}° (the model's field)",
        f"best field {best}°: {errs[best]:.1f} px",
    ]
