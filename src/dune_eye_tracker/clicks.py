"""Clicks: heard from dune, learnt from, and logged.

`run` hears every click from `dune events pointer` (`Clicks`). When you click you look at
the pointer (WebGazer does the same): the camera's frames from `BEFORE` to `AFTER` s before
it (the eyes there already, the hand not yet), when the gaze was steady (`STEADY` px),
become a calibration point (`gaze.GazeLearner`), however far from the click it guessed;
every `REFIT` clicks kept a new fit, used when it does no worse (`MARGIN`) on the last
`CHECK`. Not all clicks count the same (`WEIGHTS`).

Every click is also a line of `clicks.jsonl` (`log`), never cut and kept across
calibrations: where it was, the guess, what made it and what the tracker knew then (how the
pointer got there, the tile, what could be clicked near), for a model of where clicks go.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

BEFORE, AFTER = 0.35, 0.05  # s before the click: the frames between
STEADY = 100.0  # px: the spread of the guesses there, at most
REFIT = 5  # clicks kept between fits
# clicks a new fit is checked on (half of them while there are fewer): 5 were noise, the
# median miss over 5 clicks spanning 124..305 px, over 50 about ±25 px
CHECK = 50
# the fit in use was fitted on most of those clicks, which flatters it (1..13% on the
# user's clicks): a new fit up to this much worse on them is still taken
MARGIN = 1.15
# every click, one line, never cut: the click model's data (`log`)
LOG = "clicks.jsonl"
MIN_SAMPLES = 3
# how much a click counts in a fit, by what made it (dune's `by`): the mouse and the voice
# (`command`: "click", `name`: "click on X", dune-a11y) count the same; right Ctrl+Shift
# lands where the gaze already put the pointer (it partly teaches it its own guess). A
# click without (an older dune): 1.
WEIGHTS = {"mouse": 1.0, "key": 0.5, "command": 1.0, "name": 1.0}


def weight(by: str | None) -> float:
    return WEIGHTS.get(by, 1.0) if by is not None else 1.0


def by_hand(by: str | None) -> bool:
    """Whether the hand put the pointer there, so the click says where the gaze was off."""
    return by is None or by == "mouse"


def log(path: Path, line: dict):
    """One line of `clicks.jsonl` (`path`), the time to the millisecond first."""
    try:
        with open(path, "a") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="milliseconds")} | line) + "\n")
    except OSError:  # a full disk must not stop the pointer
        pass


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


def how(event: dict) -> dict:
    """What made a `dune events pointer` click (`by`: mouse, key, command; none from an
    older dune) and which button."""
    return {k: event[k] for k in ("by", "button") if k in event}


def zoom_of(event: dict) -> dict | None:
    """A `zoom` `in` line: center, scale and area (monitor coordinates); None for `out`."""
    if event.get("change") != "in":
        return None
    keys = ("x", "y", "scale", "area_x", "area_y", "area_w", "area_h")
    return {k: event[k] for k in keys} if all(k in event for k in keys) else None


class Focus:
    """The focused tile, from `dune events window,strip`: which (`tile`), the one before
    it (`prev`), how long ago it changed (`age`, s) and where it is on its monitor (`x`,
    `y`, `w`, `h`, from the strip's layout; absent when it is not on that monitor's
    strip). Only numbers: no titles nor apps."""

    def __init__(self):
        self.tile: int | None = None
        self.prev: int | None = None
        self.since = 0.0
        self.strips: dict[str, dict] = {}  # output: its last `layout` line

    def apply(self, e: dict, t: float):
        if e.get("event") == "window":
            tile = e.get("tile")
            if e.get("change") == "focus" and tile != self.tile:
                self.prev, self.tile, self.since = self.tile, tile, t
            elif e.get("change") == "close" and tile == self.tile:
                self.prev, self.tile, self.since = self.tile, None, t
        elif e.get("event") == "strip" and e.get("change") == "layout" and "output" in e:
            self.strips[e["output"]] = e

    def at(self, output: str, t: float) -> dict | None:
        """The focus as a click on `output` at `t` finds it; None when nothing has it."""
        if self.tile is None:
            return None
        out = {"tile": self.tile, "prev": self.prev, "age": round(max(t - self.since, 0.0), 3)}
        strip = self.strips.get(output) or {}
        for tile in strip.get("tiles", []):
            if tile.get("tile") == self.tile:
                out |= {"x": tile["x"] - strip.get("view", 0), "y": tile["y"], "w": tile["w"], "h": tile["h"]}
        return out


class Clicks(threading.Thread):
    """`dune events pointer,zoom,window,strip`, in the background: ("click", time heard,
    x, y, output, the tile it was on, `how` it was made and the `focus` then) of each left
    or right click on a queue, (x, y)
    where it was seen; ("move", ..., None, None) as the mouse moves the pointer; ("zoom",
    t, 0, 0, output, the zoom or None, None) as dune's zoom goes in or out."""

    def __init__(self):
        super().__init__(daemon=True)
        self.queue: Queue = Queue()
        self.proc: subprocess.Popen | None = None
        self.focus = Focus()

    def run(self):
        try:
            self.proc = subprocess.Popen(["dune", "events", "pointer,zoom,window,strip"], stdout=subprocess.PIPE, text=True)
        except OSError:
            return
        for line in self.proc.stdout:
            t = time.monotonic()
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "zoom" and "output" in e:
                self.queue.put(("zoom", t, 0, 0, e["output"], zoom_of(e), None))
                continue
            if e.get("event") in ("window", "strip"):
                self.focus.apply(e, t)
                continue
            if e.get("event") != "pointer":
                continue
            if e.get("change") == "move":
                self.queue.put(("move", t, e["x"], e["y"], e["output"], None, None))
            elif e.get("button") in ("left", "right"):
                focus = self.focus.at(e["output"], t)
                made = how(e) | ({"focus": focus} if focus else {})
                self.queue.put(("click", t, *seen(e)[:2], e["output"], seen(e)[2], made))

    def pending(self):
        while True:
            try:
                yield self.queue.get_nowait()
            except Empty:
                return

    def stop(self):
        if self.proc:
            self.proc.terminate()
