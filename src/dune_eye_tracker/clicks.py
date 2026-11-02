"""Learning from clicks: when you click, you look at the pointer (WebGazer does the same).

`run` hears every click from `dune events pointer`. For one on the calibrated monitor it
takes the camera's samples from `BEFORE` to `AFTER` s before it (the eyes there already,
the hand not yet), and keeps them when the gaze was steady (`STEADY` px), however far
from the click it guessed: you look at the pointer when you click (the user does so on
purpose), and the clicks it is furthest off on are the ones it learns most from. Every `REFIT` clicks kept, a
`model.HeadFix` with the gaze terms is fitted on the head calibration's samples and the
clicks (the last `KEEP`), and used when it does no worse than the one in use on the last
`REFIT` clicks, fitted without them. Clicks and fix go in calibration.json
(`click_samples`, `click_fix`); a new `calibrate` drops them.
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
REFIT = 5  # clicks kept between fits, and checked on
KEEP = 400  # clicks kept at most (the last)
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

    def guess(self, rows: list[dict]) -> np.ndarray:
        g = self.eyes.predict(rows)
        return self.fix.correct(g, rows) if self.fix else g

    def seen(self, t: float, feats: dict):
        self.history.append((t, feats))

    def click(self, t: float, x: int, y: int, output: str) -> str:
        """A click at `t` (this clock): what came of it ("kept", or why not)."""
        if output != self.output:
            return "other monitor"
        rows = [f for (ts, f) in self.history if t - BEFORE <= ts <= t - AFTER and f["open"] >= BLINK]
        if len(rows) < MIN_SAMPLES:
            self.skipped += 1
            return "no face before it"
        g = self.guess(rows)
        mid = np.median(g, axis=0)
        spread = float(np.sqrt((np.linalg.norm(g - mid, axis=1) ** 2).mean()))
        if spread > STEADY:
            self.skipped += 1
            return f"gaze moving ({spread:.0f} px)"
        far = float(np.linalg.norm(mid - (x, y)))
        w, h = self.size
        self.samples += [{"point": self.next_id, "target": [x / w, y / h], "feats": f} for f in rows]
        self.next_id += 1
        self.kept += 1
        self.since_fit += 1
        ids = sorted({s["point"] for s in self.samples})
        if len(ids) > KEEP:
            old = set(ids[: len(ids) - KEEP])
            self.samples = [s for s in self.samples if s["point"] not in old]
        return f"kept ({far:.0f} px off)"

    def arrays(self, samples: list[dict]):
        w, h = self.size
        rows = [s["feats"] for s in samples]
        targets = np.array([[s["target"][0] * w, s["target"][1] * h] for s in samples])
        return rows, targets, np.array([s["point"] for s in samples])

    def refit(self) -> str | None:
        """Every `REFIT` clicks kept: a new fix, when it is no worse (what came of it)."""
        if self.since_fit < REFIT:
            return None
        ids = sorted({s["point"] for s in self.samples})
        recent = set(ids[-REFIT:])
        head = list(self.data.get("head_samples", []))
        if not head and len(ids) < 2 * REFIT:
            return None  # nothing to fit on yet besides the clicks it is checked on
        self.since_fit = 0
        train = head + [s for s in self.samples if s["point"] not in recent]
        test = [s for s in self.samples if s["point"] in recent]
        ref = self.data.get("head") or dict.fromkeys(model.HEAD_COLUMNS, 0.0)
        candidate, _, _ = model.fit_head(self.eyes, *self.arrays(train), ref, self.size)
        rows, targets, groups = self.arrays(test)
        now = model.error(self.guess(rows), targets, groups)
        new = model.error(candidate.correct(self.eyes.predict(rows), rows), targets, groups)
        note = f"{len(ids)} clicks: on the last {REFIT}, {now:.0f} px off now, {new:.0f} with the new fit"
        if new <= now:
            self.fix, _, _ = model.fit_head(self.eyes, *self.arrays(head + self.samples), ref, self.size)
            self.data["click_fix"] = self.fix.to_json() | {"clicks": len(ids), "before": now, "after": new}
            note += ": used"
        else:
            note += ": kept the old one"
        self.save()
        return note

    def save(self):
        self.data["click_samples"] = self.samples
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)


class Clicks(threading.Thread):
    """`dune events pointer`, in the background: ("click", time heard, x, y, output) of
    each left or right click on a queue, and ("move", ...) as the mouse moves the pointer."""

    def __init__(self):
        super().__init__(daemon=True)
        self.queue: Queue = Queue()
        self.proc: subprocess.Popen | None = None

    def run(self):
        try:
            self.proc = subprocess.Popen(["dune", "events", "pointer"], stdout=subprocess.PIPE, text=True)
        except OSError:
            return
        for line in self.proc.stdout:
            t = time.monotonic()
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") != "pointer":
                continue
            if e.get("change") == "move":
                self.queue.put(("move", t, e["x"], e["y"], e["output"]))
            elif e.get("button") in ("left", "right"):
                self.queue.put(("click", t, e["x"], e["y"], e["output"]))

    def pending(self):
        while True:
            try:
                yield self.queue.get_nowait()
            except Empty:
                return

    def stop(self):
        if self.proc:
            self.proc.terminate()
