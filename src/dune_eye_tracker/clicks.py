"""Learning from clicks: when you click, you look at the pointer (WebGazer does the same).

`run` hears every click from `dune events pointer`. For one on the calibrated monitor it
takes the camera's samples from `BEFORE` to `AFTER` s before it (the eyes there already,
the hand not yet), and keeps them when the gaze was steady (`STEADY` px), however far
from the click it guessed: you look at the pointer when you click (the user does so on
purpose), and the clicks it is furthest off on are the ones it learns most from. Every `REFIT` clicks kept, a
`model.HeadFix` with the gaze terms is fitted on the head calibration's samples and the
clicks (the last `KEEP`), and used when it does no worse (`MARGIN`) than the one in use on
the last `CHECK` clicks, fitted without them. `run` fits in a thread (`refit_background`):
on 400 clicks a fit took 0.7 s, during which the pointer stood still. Clicks and fix go in
calibration.json (`click_samples`, `click_fix`); a new `calibrate` drops them.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from . import model
from .features import BLINK

BEFORE, AFTER = 0.35, 0.05  # s before the click: the samples between
STEADY = 100.0  # px: the spread of the guesses there, at most
REFIT = 5  # clicks kept between fits
# clicks a new fit is checked on (half of them while there are fewer): 5 were noise, the
# median miss over 5 clicks spanning 124..305 px, over 50 about ±25 px
CHECK = 50
# the fix in use was fitted on most of those clicks (it saw 45 of 50), which flatters it by
# 1..13% on the user's clicks: a new fit up to this much worse on them is still taken
MARGIN = 1.15
# clicks kept at most (the last): fitted on the last N before 100 it had not seen, the miss
# was 260 / 220 / 194 px with 100 / 200 / 300, still going down at 400
KEEP = 1000
# every click, one line, never cut: the click model's data (`ClickLearner.log`)
LOG = "clicks.jsonl"
MIN_SAMPLES = 3


class ClickLearner:
    def __init__(self, data: dict, eyes: model.Model, fix: model.HeadFix | None, path: Path):
        self.data, self.eyes, self.fix, self.path = data, eyes, fix, path
        self.size = tuple(data["size"])
        self.output = data.get("output")
        self.samples: list[dict] = list(data.get("click_samples", []))
        self.next_id = 1 + max((s["point"] for s in self.samples), default=10_000)
        self.since_fit = 0
        self.history: deque = deque(maxlen=90)  # (t, feats): the last 3 s
        self.kept = self.skipped = 0
        self.job: _Fit | None = None  # the fit going in the background

    def guess(self, rows: list[dict]) -> np.ndarray:
        g = self.eyes.predict(rows)
        return self.fix.correct(g, rows) if self.fix else g

    def seen(self, t: float, feats: dict):
        self.history.append((t, feats))

    def click(self, t: float, x: int, y: int, output: str, ctx: dict | None = None) -> str:
        """A click at `t` (this clock): what came of it ("kept", or why not). `ctx`, from
        dune, is the tile it was on (`context`): kept with it, for learning where clicks go
        inside each app's window."""
        if output != self.output:
            return "other monitor"
        rows = [f for (ts, f) in self.history if t - BEFORE <= ts <= t - AFTER and f["open"] >= BLINK]
        if len(rows) < MIN_SAMPLES:
            self.skipped += 1
            return "no face before it"
        g = self.guess(rows)
        mid = np.median(g, axis=0)
        spread = float(np.sqrt((np.linalg.norm(g - mid, axis=1) ** 2).mean()))
        self.log(x, y, rows, mid, spread, ctx)
        if spread > STEADY:
            self.skipped += 1
            return f"gaze moving ({spread:.0f} px)"
        far = float(np.linalg.norm(mid - (x, y)))
        w, h = self.size
        extra = {"ctx": ctx} if ctx else {}
        self.samples += [{"point": self.next_id, "target": [x / w, y / h], "feats": f} | extra for f in rows]
        self.next_id += 1
        self.kept += 1
        self.since_fit += 1
        ids = sorted({s["point"] for s in self.samples})
        if len(ids) > KEEP:
            old = set(ids[: len(ids) - KEEP])
            self.samples = [s for s in self.samples if s["point"] not in old]
        return f"kept ({far:.0f} px off)"

    def log(self, x: int, y: int, rows: list[dict], guess, spread: float, ctx: dict | None):
        """One line per click in `clicks.jsonl` beside calibration.json, never cut and kept
        across calibrations: the data of a model of where clicks go (docs/click-model.md),
        for as long as more of it still helps. The eyes' numbers (the median of the frames
        before it) are kept, not only the guess, so a guess can be made again with another
        calibration; a moving gaze too, marked (`spread`)."""
        feats = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        line = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "calibration": self.data.get("time"),
            "output": self.output,
            "size": list(self.size),
            "seen": [x, y],
            "guess": [round(float(guess[0]), 1), round(float(guess[1]), 1)],
            "spread": round(spread, 1),
            "feats": feats,
        } | ({"ctx": ctx} if ctx else {})
        try:
            with open(self.path.with_name(LOG), "a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:  # a full disk must not stop the pointer
            pass

    def arrays(self, samples: list[dict]):
        w, h = self.size
        rows = [s["feats"] for s in samples]
        targets = np.array([[s["target"][0] * w, s["target"][1] * h] for s in samples])
        return rows, targets, np.array([s["point"] for s in samples])

    def due(self) -> tuple | None:
        """Every `REFIT` clicks kept: what a fit is made from (the clicks as they are now,
        the fix in use), else None."""
        if self.since_fit < REFIT:
            return None
        ids = {s["point"] for s in self.samples}
        head = list(self.data.get("head_samples", []))
        if not head and len(ids) < 2 * REFIT:
            return None  # nothing to fit on yet besides the clicks it is checked on
        self.since_fit = 0
        ref = self.data.get("head") or dict.fromkeys(model.HEAD_COLUMNS, 0.0)
        return list(self.samples), self.fix, head, ref

    def fit(self, samples: list[dict], fix, head: list[dict], ref: dict):
        """A new fix from `samples`, checked against `fix` on the last `CHECK` clicks:
        (the new fix or None to keep `fix`, what to save with it, what came of it). Reads
        nothing that changes meanwhile: it runs in a thread."""
        ids = sorted({s["point"] for s in samples})
        checked = min(CHECK, max(REFIT, len(ids) // 2))
        recent = set(ids[-checked:])
        train = head + [s for s in samples if s["point"] not in recent]
        test = [s for s in samples if s["point"] in recent]
        candidate, _, _ = model.fit_head(self.eyes, *self.arrays(train), ref, self.size)
        rows, targets, groups = self.arrays(test)
        guess = self.eyes.predict(rows)
        now = model.error(fix.correct(guess, rows) if fix else guess, targets, groups)
        new = model.error(candidate.correct(guess, rows), targets, groups)
        note = f"{len(ids)} clicks: on the last {checked}, {now:.0f} px off now, {new:.0f} with the new fit"
        if new > now * MARGIN:
            return None, None, note + ": kept the old one"
        full, _, _ = model.fit_head(self.eyes, *self.arrays(head + samples), ref, self.size)
        return full, {"clicks": len(ids), "before": now, "after": new}, note + ": used"

    def apply(self, result) -> str:
        fix, extra, note = result
        if fix is not None:
            self.fix = fix
            self.data["click_fix"] = fix.to_json() | extra
        self.save()
        return note

    def refit(self) -> str | None:
        """A fit now, when one is due (what came of it)."""
        job = self.due()
        return None if job is None else self.apply(self.fit(*job))

    def refit_background(self) -> str | None:
        """`refit` in a thread, the pointer not waiting for it: called every frame, it
        starts a fit when one is due and, once it is done, puts it in use (what came of it)."""
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
        self.data["click_samples"] = self.samples
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)


class _Fit(threading.Thread):
    def __init__(self, learner: ClickLearner, job: tuple):
        super().__init__(daemon=True)
        self.learner, self.job = learner, job
        self.result = (None, None, "fit failed")

    def run(self):
        try:
            self.result = self.learner.fit(*self.job)
        except Exception as e:  # a bad fit must not stop the pointer
            self.result = (None, None, f"fit failed: {e}")


# what dune says of the tile a click was on (a dune from 2026-11-02 on)
CONTEXT = ("tile", "app_id", "tile_x", "tile_y", "tile_w", "tile_h")


def context(event: dict) -> dict | None:
    """The tile a `dune events pointer` click was on: its app and where in it; None off a
    tile (or an older dune)."""
    if not all(k in event for k in CONTEXT):
        return None
    return {k: event[k] for k in CONTEXT}


def seen(event: dict) -> tuple[int, int, dict | None]:
    """Where a click was seen, and its context: while dune's zoom was in (`zoom_x`...) the
    eyes were on the magnified point, not on the real one (x, y), which goes in the context
    with the scale; else (x, y) itself."""
    ctx = context(event)
    if "zoom_x" not in event:
        return event["x"], event["y"], ctx
    zoomed = {"real_x": event["x"], "real_y": event["y"], "zoom_scale": event.get("zoom_scale")}
    return event["zoom_x"], event["zoom_y"], (ctx or {}) | zoomed


def zoom_of(event: dict) -> dict | None:
    """A `zoom` `in` line: center, scale and area (monitor coordinates); None for `out`."""
    if event.get("change") != "in":
        return None
    keys = ("x", "y", "scale", "area_x", "area_y", "area_w", "area_h")
    return {k: event[k] for k in keys} if all(k in event for k in keys) else None


class Clicks(threading.Thread):
    """`dune events pointer,zoom`, in the background: ("click", time heard, x, y, output,
    the tile it was on) of each left or right click on a queue, (x, y) where it was seen;
    ("move", ..., None) as the mouse moves the pointer; ("zoom", t, 0, 0, output, the zoom
    or None) as dune's zoom goes in or out."""

    def __init__(self):
        super().__init__(daemon=True)
        self.queue: Queue = Queue()
        self.proc: subprocess.Popen | None = None

    def run(self):
        try:
            self.proc = subprocess.Popen(["dune", "events", "pointer,zoom"], stdout=subprocess.PIPE, text=True)
        except OSError:
            return
        for line in self.proc.stdout:
            t = time.monotonic()
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "zoom" and "output" in e:
                self.queue.put(("zoom", t, 0, 0, e["output"], zoom_of(e)))
                continue
            if e.get("event") != "pointer":
                continue
            if e.get("change") == "move":
                self.queue.put(("move", t, e["x"], e["y"], e["output"], None))
            elif e.get("button") in ("left", "right"):
                self.queue.put(("click", t, *seen(e)[:2], e["output"], seen(e)[2]))

    def pending(self):
        while True:
            try:
                yield self.queue.get_nowait()
            except Empty:
                return

    def stop(self):
        if self.proc:
            self.proc.terminate()
