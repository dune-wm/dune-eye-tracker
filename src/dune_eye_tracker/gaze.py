"""Where you look on the screen: the eyes put in the head in 3D, the iris found in the full
1080p frame (its dark centroid), the visual axis met with the camera's plane, then a
quadratic map to the screen (`models.Geometric`). Live it was 114-124 px off (median; the
2D eyes model it replaced, ~180 on clicks and 360-500 with the head free).

The calibration is a recording, `gaze.npz` (numbers only: landmarks, pose, the iris found;
no images): the 28 points `calibrate` showed (or a recording's, `calibrate --from`), and the
clicks kept in use (`click` block), which keep it fitted as you sit and light change.

Kept up by use, with no calibration to sit through (`GazeLearner`):
- every click on its monitor with the gaze steady before it adds its frames as a point
  (you look at the pointer when you click), counting by what made it (`clicks.WEIGHTS`);
- the last `SHIFT_CLICKS` mouse clicks give the shift taken off every guess, vertically
  (`SHIFT_AXES`): sitting differently moved the gaze ~130 px down in one session (13:38),
  and 9 points took it back to ~121 px in replay;
- every `REFIT` clicks a new fit on the calibration and the clicks (the last `KEEP`), in a
  thread, used when no worse (`MARGIN`) than the one in use on the last `CHECK` clicks,
  fitted without them.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np
from . import evaluate
from .camera import BLINK
from .clicks import AFTER, BEFORE, CHECK, LOG, MARGIN, MIN_SAMPLES, REFIT, STEADY, by_hand, log, weight
from .models import Geometric
from .recording import BLOCKS, Recording, state_dir

SOURCE = "dark"  # the iris's dark centroid (in dune-gaze the steadiest across sessions)
QUADRATIC_FROM = 24  # points from which the map gets its bend (fewer: affine)
KEEP = 300  # clicks kept at most (the last): ~9 frames each, a fit on all of them ~1 s
SHIFT_CLICKS = 9  # the last mouse clicks the shift comes from
SHIFT_FROM = 5  # mouse clicks it takes to have one
SHIFT_AXES = (False, True)  # the shift's axes taken off (x, y): y only (dune-gaze replay)
CLICK = BLOCKS.index("click")
# Turning the head to one side further than looking at the screen takes it steps the strip
# a tile (`track`): how far is past the 1% most turned of the clicks by `STEP_MARGIN` (with
# `STEP_CLICKS` of them), else `STEP_TURN` from the calibration's head
STEP_TURN = 0.35
STEP_MARGIN = 0.12
STEP_CLICKS = 200


def store_path() -> Path:
    return state_dir() / "gaze.npz"


def frames(rec: Recording, blocks=("calibration", "click")) -> np.ndarray:
    """The frames a fit takes: each block's own window, the eyes open."""
    parts = [evaluate.frames(rec, b) for b in blocks if np.any(rec.block == BLOCKS.index(b))]
    return np.concatenate(parts) if parts else np.zeros(0, int)


def weights(rec: Recording, idx: np.ndarray) -> np.ndarray:
    """How much each frame counts: a click's by what made it (`meta.by`), the rest 1."""
    by = rec.meta.get("by") or {}
    return np.array([weight(by.get(str(p))) if b == CLICK else 1.0 for b, p in zip(rec.block[idx], rec.point[idx])])


def fit(rec: Recording, idx: np.ndarray | None = None) -> Geometric:
    idx = frames(rec) if idx is None else idx
    groups = evaluate.groups(rec, idx)
    points = len(np.unique(groups))
    if points < 6:
        raise ValueError(f"{points} points to fit on: too few")
    order = 2 if points >= QUADRATIC_FROM else 1
    model = Geometric(f"geometric{', quadratic' if order == 2 else ''} [{SOURCE}]", order, source=SOURCE)
    return model.fit(rec, idx, groups, weights(rec, idx))


def miss(model: Geometric, rec: Recording, idx: np.ndarray) -> float:
    """The median miss over the points of frames `idx` (each point's median guess)."""
    g, groups = model.predict(rec, idx), evaluate.groups(rec, idx)
    return float(np.median([np.linalg.norm(np.median(g[groups == u], axis=0) - rec.target[idx][groups == u][0]) for u in np.unique(groups)]))


def one(samples, screen) -> Recording | None:
    """Camera frames (`camera.Frame`) as a recording the model takes; None when none
    has the face and the iris."""
    ok = [s for s in samples if s.landmarks is not None and s.matrix is not None and s.eyes is not None]
    if not ok:
        return None
    n = len(ok)
    return Recording(
        np.array([s.landmarks for s in ok], float), np.array([s.matrix for s in ok], float), np.array([s.t for s in ok]),
        np.zeros(n, int), np.zeros(n, int), np.zeros((n, 2)), np.zeros(n), {"screen": list(screen), "image": list(ok[0].image)}, "live",
        limbus=np.array([s.eyes["limbus"] for s in ok], float), dark=np.array([s.eyes["dark"] for s in ok], float),
        maps=np.array([s.eyes["maps"] for s in ok], float),
    )


class Gaze:
    """The model fitted on `rec` (`gaze.npz`), and the shift taken off its guesses."""

    def __init__(self, rec: Recording, model: Geometric | None = None):
        self.rec = rec
        self.screen = rec.screen
        self.output = rec.meta.get("output")
        self.time = rec.meta.get("time")
        self.model = model or fit(rec)
        self.offset = np.zeros(2)

    def head_ref(self) -> tuple[float, float]:
        """Where the face pointed while calibrating (`head_x`, `head_y`, the median)."""
        m = self.rec.matrix[self.rec.block == BLOCKS.index("calibration")]
        return (float(np.median(m[:, 0, 2])), float(np.median(m[:, 1, 2]))) if len(m) else (0.0, 0.0)

    def step_turns(self) -> tuple[float, float]:
        """`head_x` past which the head steps the strip left / right: past the 1% most turned
        of the clicks (looking at the screen) by `STEP_MARGIN`, else `STEP_TURN` from the
        calibration's head."""
        rec = self.rec
        click = rec.block == CLICK
        ids = np.unique(rec.point[click])
        if len(ids) >= STEP_CLICKS:
            xs = [float(np.median(rec.matrix[click & (rec.point == i), 0, 2])) for i in ids]
            return float(np.percentile(xs, 99)) + STEP_MARGIN, float(np.percentile(xs, 1)) - STEP_MARGIN
        ref = self.head_ref()[0]
        return ref + STEP_TURN, ref - STEP_TURN

    def model_id(self) -> str:
        """Which gaze a game was played with: the calibration's time, the clicks it had (the
        shift comes from them), and a hash of the fit (refitted every few clicks)."""
        import hashlib

        h = hashlib.sha1(np.concatenate([np.ravel(self.model.p), np.ravel(self.model.w)]).tobytes()).hexdigest()[:8]
        return f"{self.time}/{len(np.unique(self.rec.point[self.rec.block == CLICK]))}/{h}"

    def predict(self, samples) -> np.ndarray | None:
        """The gaze (screen px, not kept on it: off it is how looking away shows) for each
        sample with the face and the iris; None when none has."""
        r = one(samples, self.screen)
        if r is None:
            return None
        return self.model.predict(r, np.arange(len(r.t))) - self.offset


def concat(a: Recording, b: Recording) -> Recording:
    """`a` then `b`'s frames, `a`'s meta."""
    cat = lambda k: np.concatenate([getattr(a, k), getattr(b, k)])  # noqa: E731
    return Recording(*(cat(k) for k in ("landmarks", "matrix", "t", "block", "point", "target", "since")), dict(a.meta), a.name,
                     limbus=cat("limbus"), dark=cat("dark"), maps=cat("maps"))


def subset(rec: Recording, keep: np.ndarray) -> Recording:
    return Recording(rec.landmarks[keep], rec.matrix[keep], rec.t[keep], rec.block[keep], rec.point[keep], rec.target[keep],
                     rec.since[keep], dict(rec.meta), rec.name, limbus=rec.limbus[keep], dark=rec.dark[keep], maps=rec.maps[keep])


class GazeLearner:
    """The gaze kept fitted by the clicks (module doc). `seen` every camera sample, `click`
    every click, `refit_background` every frame; `save` at the end."""

    def __init__(self, g: Gaze, path: Path | None = None):
        path = path or store_path()
        self.g, self.path = g, path
        self.history: deque = deque(maxlen=90)  # the last 3 s of samples with the iris
        clicks = g.rec.point[g.rec.block == CLICK]
        self.next_id = int(clicks.max()) + 1 if len(clicks) else 0
        self.since_fit = 0
        self.kept = 0
        self.job: _Fit | None = None
        self.shift_note = self.update_shift()

    def seen(self, sample):
        if sample.landmarks is not None and sample.eyes is not None and sample.feats is not None:
            self.history.append(sample)

    def click(self, t: float, x: int, y: int, output: str | None, by: str | None = None, line: dict | None = None) -> str:
        """A click at `t` (this clock): its frames added as a point (what came of it). It
        goes to `clicks.jsonl` too, with `line` (what made it, the tile, how the pointer got
        there...): where, the guess (the shift taken off, as the pointer had it), how steady,
        the head, what came of it."""
        if output != self.g.output:
            return "other monitor"
        rows = [s for s in self.history if t - BEFORE <= s.t <= t - AFTER and s.feats["open"] >= BLINK]
        record = {"calibration": self.g.time, "output": output, "size": list(self.g.screen), "seen": [x, y]} | (line or {})
        if len(rows) < MIN_SAMPLES:
            log(self.path.with_name(LOG), record | {"said": "no iris before it"})
            return "no iris before it"
        r = one(rows, self.g.screen)
        guess = self.g.model.predict(r, np.arange(len(r.t)))
        mid = np.median(guess, axis=0)
        spread = float(np.sqrt((np.linalg.norm(guess - mid, axis=1) ** 2).mean()))
        record |= {
            "guess": [round(float(v), 1) for v in mid - self.g.offset], "spread": round(spread, 1),
            "feats": {k: float(np.median([s.feats[k] for s in rows])) for k in rows[0].feats},
        }
        if spread > STEADY:
            log(self.path.with_name(LOG), record | {"said": f"gaze moving ({spread:.0f} px)"})
            return f"gaze moving ({spread:.0f} px)"
        n = len(r.t)
        r.block, r.point = np.full(n, CLICK), np.full(n, self.next_id)
        r.target, r.since = np.tile([float(x), float(y)], (n, 1)), r.t - t
        rec = concat(self.g.rec, r)
        rec.meta["by"] = dict(rec.meta.get("by") or {}) | {str(self.next_id): by}
        ids = np.unique(rec.point[rec.block == CLICK])
        if len(ids) > KEEP:  # the oldest clicks go
            rec = subset(rec, ~((rec.block == CLICK) & np.isin(rec.point, ids[: len(ids) - KEEP])))
        self.g.rec = rec
        self.next_id += 1
        self.kept += 1
        self.since_fit += 1
        off = mid - self.g.offset - (x, y)
        said = f"kept ({np.linalg.norm(off):.0f} px off)"
        log(self.path.with_name(LOG), record | {"said": said})
        if by_hand(by):
            self.shift_note = self.update_shift()
        return said

    def mouse_clicks(self) -> np.ndarray:
        """The last `SHIFT_CLICKS` clicks made by the mouse, their ids."""
        by = self.g.rec.meta.get("by") or {}
        ids = np.unique(self.g.rec.point[self.g.rec.block == CLICK])
        return np.array([i for i in ids if by_hand(by.get(str(i)))][-SHIFT_CLICKS:], dtype=int)

    def update_shift(self) -> str:
        """The shift from the last mouse clicks, with the model in use (none with fewer
        than `SHIFT_FROM`)."""
        ids = self.mouse_clicks()
        if len(ids) < SHIFT_FROM:
            self.g.offset = np.zeros(2)
            return f"no shift yet ({len(ids)} of {SHIFT_FROM} mouse clicks)"
        rec = self.g.rec
        idx = np.flatnonzero((rec.block == CLICK) & np.isin(rec.point, ids))
        guess, groups = self.g.model.predict(rec, idx), rec.point[idx]
        d = np.array([np.median(guess[groups == u], axis=0) - rec.target[idx][groups == u][0] for u in ids])
        self.g.offset = np.where(SHIFT_AXES, np.median(d, axis=0), 0.0)
        return f"shift {self.g.offset[1]:+.0f} px vertically from the last {len(ids)} mouse clicks"

    def due(self) -> tuple | None:
        if self.since_fit < REFIT:
            return None
        ids = np.unique(self.g.rec.point[self.g.rec.block == CLICK])
        if len(ids) < 2 * REFIT:
            return None
        self.since_fit = 0
        return self.g.rec, self.g.model

    @staticmethod
    def fit(rec: Recording, model: Geometric):
        """A new model on `rec`, checked against `model` on the last `CHECK` clicks: (the
        new one or None, what came of it). Reads nothing that changes: it runs in a thread."""
        ids = np.unique(rec.point[rec.block == CLICK])
        checked = min(CHECK, max(REFIT, len(ids) // 2))
        recent = (rec.block == CLICK) & np.isin(rec.point, ids[-checked:])
        all_idx = frames(rec)
        train, test = all_idx[~recent[all_idx]], all_idx[recent[all_idx]]
        if not len(test):
            return None, "no clicks to check a fit on"
        candidate = fit(rec, train)
        now, new = miss(model, rec, test), miss(candidate, rec, test)
        note = f"{len(ids)} clicks: on the last {checked}, {now:.0f} px off now, {new:.0f} with a new fit"
        if new > now * MARGIN:
            return None, note + ": kept the one in use"
        return fit(rec, all_idx), note + ": used"

    def apply(self, result) -> str:
        model, note = result
        if model is not None:
            self.g.model = model
            note += f"; {self.update_shift()}"
        self.save()
        return note

    def refit_background(self) -> str | None:
        """Called every frame: starts a fit when one is due, puts it in use once done."""
        if self.job is not None:
            if self.job.is_alive():
                return None
            result, self.job = self.job.result, None
            return self.apply(result)
        if (job := self.due()) is not None:
            self.job = _Fit(self, job)
            self.job.start()
        return None

    def save(self):
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        self.g.rec.save(tmp)
        tmp.replace(self.path)


class _Fit(threading.Thread):
    def __init__(self, learner: GazeLearner, job: tuple):
        super().__init__(daemon=True)
        self.learner, self.job = learner, job
        self.result = (None, "fit failed")

    def run(self):
        try:
            self.result = self.learner.fit(*self.job)
        except Exception as e:  # a bad fit must not stop the pointer
            self.result = (None, f"fit failed: {e}")


def load(path: Path | None = None) -> Gaze | None:
    path = path or store_path()
    if not path.exists():
        return None
    return Gaze(Recording.load(path))


def import_session(source: str, output: str | None, path: Path | None = None) -> int:
    """`calibrate --from REC.npz --output NAME`: a recording's calibration points (`record`,
    `live` C; dune-gaze's too) as the calibration (no clicks yet)."""
    import sys

    if not output:
        print("say the monitor it was on: --output NAME (as in dune.toml's [output.NAME])", file=sys.stderr)
        return 1
    rec = Recording.load(Path(source))
    if rec.limbus is None:
        print(f"{source} has no full-frame iris (recorded at 1280x720?): record one at 1080p", file=sys.stderr)
        return 1
    keep = rec.block == BLOCKS.index("calibration")
    if not keep.any():
        print(f"{source} has no calibration points (a session with no C?)", file=sys.stderr)
        return 1
    out = Recording(
        rec.landmarks[keep], rec.matrix[keep], rec.t[keep], rec.block[keep], rec.point[keep], rec.target[keep], rec.since[keep],
        {"screen": list(rec.screen), "image": list(rec.image), "output": output, "from": rec.name, "time": rec.meta.get("time") or rec.name}, "gaze",
        limbus=rec.limbus[keep], dark=rec.dark[keep], maps=rec.maps[keep],
    )
    g = Gaze(out)  # fits: a bad recording says so now, not in `run`
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    points = len(np.unique(out.point))
    print(f"calibration from {rec.name} on {output} ({rec.screen[0]}x{rec.screen[1]}, {points} points, {g.model.name}): {path}")
    return 0
