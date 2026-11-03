"""From the camera's numbers to a point on the screen: ridge regression on a polynomial.

Each feature set names the columns it takes, and which of them also get squared and
crossed terms (the eyes move the gaze non-linearly towards the edges). The ridge
strength is chosen by leaving out one calibration target at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# name: (columns, columns with quadratic terms too)
SETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "eyes": (("eye_u", "eye_v"), ("eye_u", "eye_v")),
    "head": (("head_x", "head_y", "pos_x", "pos_y"), ("head_x", "head_y")),
    "both": (("eye_u", "eye_v", "head_x", "head_y", "pos_x", "pos_y"), ("eye_u", "eye_v")),
}

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def design(x: np.ndarray, quad: list[int]) -> np.ndarray:
    """1, the columns of `x`, and the products of each pair (with itself) of `quad`."""
    cols = [np.ones(len(x)), *x.T]
    for i, a in enumerate(quad):
        for b in quad[i:]:
            cols.append(x[:, a] * x[:, b])
    return np.column_stack(cols)


def ridge(a: np.ndarray, y: np.ndarray, lam: float, w: np.ndarray | None = None) -> np.ndarray:
    """Least squares with `lam` on every weight but the intercept's; with `w`, each row
    counts that much."""
    reg = lam * np.eye(a.shape[1])
    reg[0, 0] = 0.0
    aw = a if w is None else a * w[:, None]
    return np.linalg.solve(aw.T @ a + reg, aw.T @ y)


@dataclass
class Model:
    columns: tuple[str, ...]
    quad: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    lam: float

    def matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        x = np.array([[r[c] for c in self.columns] for r in rows], dtype=float)
        x = (x - self.mean) / self.scale
        return design(x, [self.columns.index(c) for c in self.quad])

    def predict(self, rows: list[dict[str, float]], bounds: tuple[float, float] | None = None) -> np.ndarray:
        """(n, 2) screen points; with `bounds` (w, h), kept on the screen (a feature far
        from any seen while calibrating would otherwise throw the point miles away)."""
        p = self.matrix(rows) @ self.weights
        if bounds is not None:
            p = np.clip(p, 0, np.array(bounds, dtype=float) - 1)
        return p


def fit(rows: list[dict[str, float]], targets: np.ndarray, groups: np.ndarray, set_name: str) -> Model:
    """A model of `set_name` from samples `rows` looking at `targets` (n, 2); `groups`
    says which calibration point each sample is of (for choosing the ridge strength)."""
    columns, quad = SETS[set_name]
    x = np.array([[r[c] for c in columns] for r in rows], dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    a = design((x - mean) / scale, [columns.index(c) for c in quad])
    best = None
    for lam in LAMBDAS:
        err = []
        for g in np.unique(groups):
            out = groups == g
            w = ridge(a[~out], targets[~out], lam)
            err.append(np.linalg.norm(a[out] @ w - targets[out], axis=1).mean())
        score = float(np.mean(err))
        if best is None or score < best[0]:
            best = (score, lam)
    lam = best[1]
    return Model(columns, quad, mean, scale, ridge(a, targets, lam), lam)


@dataclass
class Report:
    """How far off a model is on the test points (px)."""

    median: float  # median over test points of the distance of their median prediction
    mean: float
    worst: float
    x: float  # median over test points of the horizontal part of that distance
    y: float  # and of the vertical part
    jitter: float  # mean spread of the predictions around their median, per point
    smooth: float  # the same with each prediction the mean of the last `SMOOTH` ones


SMOOTH = 8  # predictions averaged for `Report.smooth`: about 0.3 s at 27 fps


def evaluate(
    model: Model, rows: list[dict[str, float]], targets: np.ndarray, groups: np.ndarray, bounds: tuple[float, float] | None = None
) -> Report:
    pred = model.predict(rows, bounds)
    errors, xs, ys, spreads, smooth = [], [], [], [], []

    def spread(p):
        return float(np.sqrt((np.linalg.norm(p - np.median(p, axis=0), axis=1) ** 2).mean()))

    for g in np.unique(groups):
        p = pred[groups == g]
        off = np.median(p, axis=0) - targets[groups == g][0]
        errors.append(float(np.linalg.norm(off)))
        xs.append(abs(float(off[0])))
        ys.append(abs(float(off[1])))
        spreads.append(spread(p))
        if len(p) >= SMOOTH:
            kernel = np.ones(SMOOTH) / SMOOTH
            smooth.append(spread(np.column_stack([np.convolve(p[:, i], kernel, "valid") for i in (0, 1)])))
    return Report(
        float(np.median(errors)),
        float(np.mean(errors)),
        float(np.max(errors)),
        float(np.median(xs)),
        float(np.median(ys)),
        float(np.mean(spreads)),
        float(np.mean(smooth)) if smooth else float("nan"),
    )


def degrees(px: float, px_per_cm: float, distance_cm: float) -> float:
    """`px` on the screen as an angle seen from `distance_cm` away."""
    return float(np.degrees(np.arctan2(px / px_per_cm, distance_cm)))



# What the head adds to the eyes model, from where it was at the eyes' calibration.
HEAD_COLUMNS = ("head_x", "head_y", "pos_x", "pos_y")
# What a fix is fitted on now: the head, and how open the eyes are (looking down the lids
# close and the iris reads higher than it is: +0.38 with the vertical miss on 400 clicks,
# 219 -> 197 px each click left out). A fix saved before has the head only.
FIX_COLUMNS = HEAD_COLUMNS + ("open",)


@dataclass
class HeadFix:
    """gaze = eyes + bias + weights · (head - ref [, eyes' guess]): the eyes model was
    fitted with the head still; turned or moved, the eyes turn the other way to keep
    looking at the same point, and this puts back what that does to the guess. With
    `gaze_mid` (learnt from clicks, all over the screen) it also straightens the eyes'
    own guess (pulled towards the middle, say), linearly."""

    ref: np.ndarray  # (n,) the head's place (and the eyes' openness) at the eyes' calibration
    bias: np.ndarray  # (2,)
    weights: np.ndarray  # (n, 2), or (n + 2, 2) with the gaze terms
    gaze_mid: np.ndarray | None = None  # (2,) the screen's middle: gaze terms are (guess - mid) / mid
    names: tuple[str, ...] = HEAD_COLUMNS  # the n features, as `ref`

    def columns(self, eyes: np.ndarray, rows: list[dict[str, float]]) -> np.ndarray:
        d = np.array([[r[c] for c in self.names] for r in rows], dtype=float) - self.ref
        if self.gaze_mid is not None:
            d = np.column_stack([d, (eyes - self.gaze_mid) / self.gaze_mid])
        return d

    def correct(self, eyes: np.ndarray, rows: list[dict[str, float]]) -> np.ndarray:
        return eyes + self.bias + self.columns(eyes, rows) @ self.weights

    def to_json(self) -> dict:
        out = {"ref": self.ref.tolist(), "bias": self.bias.tolist(), "weights": self.weights.tolist(), "columns": list(self.names)}
        if self.gaze_mid is not None:
            out["gaze_mid"] = self.gaze_mid.tolist()
        return out

    @classmethod
    def from_json(cls, d: dict) -> "HeadFix":
        mid = np.array(d["gaze_mid"]) if d.get("gaze_mid") is not None else None
        names = tuple(d.get("columns") or HEAD_COLUMNS)  # saved before `columns`: the head only
        return cls(np.array(d["ref"]), np.array(d["bias"]), np.array(d["weights"]), mid, names)


HEAD_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def fit_head(
    eyes: Model,
    rows: list[dict[str, float]],
    targets: np.ndarray,
    groups: np.ndarray,
    ref: dict[str, float],
    size: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
):
    """A `HeadFix` on samples looking at `targets` with the head moving (with `size`, the
    screen's, the gaze terms too), and how it does on each point left out: (fix, eyes
    alone, with the fix), each the median over the points of the mean distance of a guess
    from it (px). `weights`: how much each sample counts in the fit (1 each without)."""
    names = FIX_COLUMNS
    # the calibration keeps the head's place; the openness it did not: the samples' own
    ref_v = np.array([ref[c] if c in ref else float(np.median([r[c] for r in rows])) for c in names], dtype=float)
    mid = None if size is None else np.array(size, dtype=float) / 2
    guess = eyes.predict(rows)
    residual = targets - guess
    d = HeadFix(ref_v, np.zeros(2), np.zeros((len(names), 2)), mid, names).columns(guess, rows)
    scale = d.std(axis=0)
    scale[scale < 1e-9] = 1.0
    a = np.column_stack([np.ones(len(d)), d / scale])

    def point_errors(lam):
        # per sample, not the median of a point: the head goes both ways around it, and
        # the median of guesses that swing with it would look right
        alone, fixed = [], []
        for g in np.unique(groups):
            out = groups == g
            w = ridge(a[~out], residual[~out], lam, None if weights is None else weights[~out])
            alone.append(float(np.linalg.norm(residual[out], axis=1).mean()))
            fixed.append(float(np.linalg.norm(residual[out] - a[out] @ w, axis=1).mean()))
        return alone, fixed

    best = min(HEAD_LAMBDAS, key=lambda lam: float(np.mean(point_errors(lam)[1])))
    alone, fixed = point_errors(best)
    w = ridge(a, residual, best, weights)
    fix = HeadFix(ref_v, w[0], w[1:] / scale[:, None], mid, names)
    return fix, float(np.median(alone)), float(np.median(fixed))


def error(guesses: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> float:
    """The median over the points of the mean distance of a guess from its target (px)."""
    return float(np.median([np.linalg.norm(guesses[groups == g] - targets[groups == g], axis=1).mean() for g in np.unique(groups)]))
