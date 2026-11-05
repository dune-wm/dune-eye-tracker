"""`dune-eye-tracker run`: the pointer goes where you look.

No window. The gaze (`gaze`: the eyes in the head in 3D, fitted on `calibrate`'s points and
kept fitted by the clicks) gives, each frame with a face, the iris and the eyes open, a
point on the screen; the head where it points. Both go
through a One Euro filter (steady while still, quick when they move). The pointer moves
only to where you fix your eyes: it stays put while the gaze wanders (reading, a glance:
the eyes move all the time, and a pointer glued to them is everywhere), and once the gaze
holds still (within `SETTLE` px for `SETTLE_TIME` s) more than `DEAD` px from where the
pointer last landed, it glides there: a landing, where dune-a11y is asked for a button.
Between landings the head moves it from where it landed, turned a little for the last few
pixels (MAGIC pointing, Zhai 1999); turned far to a side it steps a tile. The pointer glides there (`GLIDE`), not a bee. Without a
face, or blinking, the pointer stays. Looking off the screen (`AWAY` px past an edge
for `AWAY_TIME` s: the monitor below, the keyboard) lets go of the pointer, for the
mouse; looking back on it takes it again. Moving the mouse also takes the pointer for the
hand: it stays where the hand leaves it (head still) until the gaze moves `JUMP` px from
where it was then. Left on another monitor by the mouse, it stays there, wherever you
look, until the mouse brings it back to this one. `toggle` starts and stops it.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from OneEuroFilter import OneEuroFilter

from . import gaze as geometric
from .camera import BLINK, SHARE, Camera
from .clicks import Clicks
from .gaze import STEP_TURN
from .recording import state_dir
from .snap import Near, Puts
from .trace import LOG as TRACE_LOG
from .trace import Trace

DEAD = 120.0  # px from the last landing a fixation must be to move the pointer
# px: the gaze within this of its mean for `SETTLE_TIME` s is a fixation (40 px for 0.3 s
# was too strict live: ~8 landings a minute, the pointer felt dead)
SETTLE = 60.0
SETTLE_TIME = 0.2
HEAD_GAIN = 1200.0  # px per unit of head turn (0.05, about 3°: 60 px): slow, for the last pixels
JUMP = 300.0  # px the gaze moves from where the hand left the pointer to take it back
JUMP_TIME = 0.15  # s it stays that far first
AWAY = 150.0  # px past an edge of the screen the gaze is "off it"
AWAY_TIME = 0.3  # s it stays there before the pointer is let go
HAND_TIME = 0.5  # s after the mouse last moved during which the pointer is the hand's
# s: the pointer glides to where it goes (a jump, the head) instead of being put there,
# most of the way (63%) in this time; 0: put there
GLIDE = 0.1
# Turning the head to one side further than looking at the screen takes it (the eyes may
# stay on the screen) steps the strip one tile that way (`dune action focus_left/right`);
# the next step waits for the head to come back `STEP_BACK` towards the middle. How far
# is from the clicks (`gaze.Gaze.step_turns`). Looking left turns `head_x` up. The head
# stays turned that far `STEP_HOLD` s first: reaching for a side edge turns it nearly as
# much (on 1043 clicks, up to 0.03 short of the 1% + 0.08 it took before).
STEP_HOLD = 0.25
STEP_BACK = 0.15

# One Euro filters (Casiez et al. 2012): Hz of the cutoff when still, and how much it
# opens with speed (in the units of each: px for the gaze, the head's axis for the head).
# the gaze's beta 0.004 (was 0.01): slower to follow speed, steadier (asked in dune-gaze's live)
GAZE_FILTER = {"mincutoff": 0.6, "beta": 0.004}
HEAD_FILTER = {"mincutoff": 1.0, "beta": 3.0}
# px: the most of an edge of the screen the gaze's pointer keeps off (`usable`): dune's
# outer gap (`gap`, 0..64) and a bar's reserved band, not the empty half of a lone narrow tile
EDGE_MAX = 64
FPS = 27.0


class Pointer:
    """Where the pointer goes, from the gaze (screen px, filtered) and the head (`head_x`,
    `head_y`; looking left turns `head_x` up): the gaze lands it, the head moves it from
    there (the other way on the screen's axes) and, turned far, steps tiles."""

    def __init__(
        self,
        size: tuple[float, float],
        jump: float = JUMP,
        jump_time: float = JUMP_TIME,
        dead: float = DEAD,
        settle: float = SETTLE,
        settle_time: float = SETTLE_TIME,
        head_gain: float = HEAD_GAIN,
        away: float = AWAY,
        away_time: float = AWAY_TIME,
        head_ref=(0.0, 0.0),
        hand_time: float = HAND_TIME,
        glide: float = 0.0,
        step_at: tuple[float, float] | None = None,
        step_hold: float = STEP_HOLD,
    ):
        self.size = np.array(size, dtype=float)
        # where the gaze may put it (x0, y0, x1, y1): the screen, less dune's outer gap and
        # bars once dune's layout says where the tiles are (`usable`); the hand goes anywhere
        self.area = np.array([0.0, 0.0, self.size[0] - 1, self.size[1] - 1])
        self.jump, self.jump_time = jump, jump_time
        self.dead, self.settle, self.settle_time = dead, settle, settle_time
        self.gain = head_gain
        self.head0: np.ndarray | None = None  # the head when it landed: it moves it from there
        self.away, self.away_time = away, away_time
        self.head_ref = np.asarray(head_ref, dtype=float)
        self.away_since: float | None = None
        self.released = False  # looking off the screen: the pointer is the mouse's
        self.at: np.ndarray | None = None  # where the pointer is
        self.landed: np.ndarray | None = None  # where the last landing put it
        self.anchor: np.ndarray | None = None  # the gaze there (unclipped): `DEAD` counts from it
        self.came_from: np.ndarray | None = None  # and for the last landing (None: it was not here)
        self.recent: deque = deque(maxlen=30)  # (t, gaze) of the last frames: whether it settled
        self.far_since: float | None = None
        self.sent: tuple[int, int] | None = None
        self.hand_time = hand_time
        self.hand_until: float | None = None  # the mouse is moving the pointer until then
        self.hand_at: np.ndarray | None = None  # to there (None: on another monitor)
        self.hand_gaze: np.ndarray | None = None  # the gaze when the hand stopped
        self.elsewhere = False  # the hand left it on another monitor: left alone till it is back
        self.glide = glide
        self.shown: np.ndarray | None = None  # where the glide has it, on its way to `at`
        self.shown_t = 0.0
        # `head_x` past which the head steps the strip left / right (None: from `head_ref`)
        ref = float(self.head_ref[0])
        self.step_at = step_at or (ref + STEP_TURN, ref - STEP_TURN)
        self.step_hold = step_hold
        self.turn_since: float | None = None  # the head turned past `step_at` since then
        self.step = 0  # a tile to step this frame: -1 left, 1 right, 0 none
        self.turned = 0  # stepped that way: the head has to come back first
        # jumps so far: a late answer of where one should have landed (`snap_to`) is only
        # taken for the last one
        self.landings = 0
        # how the pointer got to `landed`, for the clicks' log (`jump`): `gaze` (a jump),
        # `back` (on the gaze, back from a head step), `hand` (the mouse), `put` (dune-a11y);
        # when; the gaze then, not kept on the screen; and how far a snap moved it
        self.land_kind: str | None = None
        self.land_t = 0.0
        self.land_gaze: np.ndarray | None = None
        self.snap_shift: np.ndarray | None = None
        self.next_kind = "gaze"
        self.hand_kind = "hand"
        self.now = 0.0
        # the last jump of the gaze as it was when the hand (or a put) took the pointer
        # from it, and its number: what the hand then put right is that jump's miss
        self.before: dict | None = None
        self.before_seq = 0

    def hand(self, t: float, xy, kind: str = "hand"):
        """The mouse moved the pointer to `xy` on this monitor (None: on another one);
        `kind` `put` when it was dune-a11y."""
        if self.land_kind in ("gaze", "back") and self.at is not None:
            self.before, self.before_seq = self.state(), self.landings
        self.hand_kind = kind
        self.hand_until = t + self.hand_time
        self.hand_at = None if xy is None else np.clip(np.asarray(xy, dtype=float), 0, self.size - 1)
        self.hand_gaze = None
        self.elsewhere = xy is None

    def hand_now(self, t: float) -> bool:
        """Whether the mouse moved the pointer just before `t` (`HAND_TIME`)."""
        return self.hand_until is not None and t < self.hand_until

    def update(self, t: float, gaze, head) -> tuple[int, int] | None:
        """The pixel to move the pointer to, or None when it stays where it is (or is let
        go). `gaze` is not kept on the screen: off it is how looking away shows."""
        gaze, head = np.asarray(gaze, dtype=float), np.asarray(head, dtype=float)
        self.step = 0
        self.now = t
        if self.elsewhere:  # on another monitor: only the mouse brings it back here
            self.at = self.sent = self.far_since = self.away_since = None
            self.hand_until = None
            return None
        if self.hand_until is not None:
            if t < self.hand_until:  # the hand's: where it put it, nothing sent
                if self.hand_at is None:
                    self.at = self.sent = None
                else:
                    self.landed, self.at, self.head0 = self.hand_at.copy(), self.hand_at.copy(), head
                    self.land_kind, self.land_t = self.hand_kind, t
                    self.land_gaze = self.snap_shift = None
                    self.sent = (int(round(self.hand_at[0])), int(round(self.hand_at[1])))
                    self.shown, self.shown_t = self.hand_at.copy(), t
                self.far_since = None
                return None
            self.hand_until, self.hand_gaze = None, gaze.copy()  # stopped: kept until you look away
        if not self.released and self.head_step(head):
            return None  # turned to step: the pointer stays (the gaze reads the turn too)
        seen = gaze  # the gaze counts the head itself: off the screen is looking away
        if np.any(seen < -self.away) or np.any(seen > self.size - 1 + self.away):
            self.away_since = t if self.away_since is None else self.away_since
            if t - self.away_since >= self.away_time:
                self.released, self.at, self.sent = True, None, None
            return None  # off for a moment (a glance) the pointer stays; longer, let go
        self.away_since = None
        if self.released:
            if np.any(seen < 0) or np.any(seen > self.size - 1):
                return None  # back only once well on the screen, not on its edge
            self.released = False
        self.recent.append((t, gaze.copy()))
        if self.hand_gaze is not None:
            if np.linalg.norm(gaze - self.hand_gaze) <= self.jump:
                self.far_since = None
                return None  # the gaze is still where it was: the pointer stays as the hand left it
            self.far_since = t if self.far_since is None else self.far_since
            if t - self.far_since < self.jump_time:
                return None
            self.hand_gaze, self.anchor, self.far_since = None, None, None  # the next fixation takes it
        elif self.at is None:
            self.land(gaze, head)
        elif (fix := self.fixation(t)) is not None and (self.anchor is None or np.linalg.norm(fix - self.anchor) > self.dead):
            self.land(fix, head)
        else:
            self.follow(head)
        if self.glide > 0 and self.shown is not None:
            k = 1.0 - np.exp(-max(t - self.shown_t, 0.0) / self.glide)
            self.shown = self.shown + (self.at - self.shown) * k
        else:
            self.shown = self.at.copy()
        self.shown_t = t
        px = (int(round(self.shown[0])), int(round(self.shown[1])))
        if px == self.sent:
            return None
        self.sent = px
        return px

    def fixation(self, t: float) -> np.ndarray | None:
        """Where the eyes are fixed: the mean of the gaze when it stayed within `settle` px
        of it the last `settle_time` s, else None."""
        if not self.recent or self.recent[0][0] > t - self.settle_time + 1e-6:
            return None  # not that long seen yet (a gap: blinking, away)
        pts = np.array([g for s, g in self.recent if s >= t - self.settle_time - 1e-6])
        if len(pts) < 0.6 * self.settle_time * FPS:
            return None  # too few frames there (blinks): not seen holding still
        mid = pts.mean(axis=0)
        return mid if np.all(np.linalg.norm(pts - mid, axis=1) <= self.settle) else None

    def land(self, gaze, head):
        """The pointer comes to rest on the gaze: a landing (dune-a11y asked, the trace)."""
        self.came_from = None if self.at is None else self.at.copy()
        self.landed = self.inside(gaze)
        self.anchor = np.array(gaze, dtype=float)
        self.land_kind, self.next_kind, self.land_t = self.next_kind, "gaze", self.now
        self.land_gaze, self.snap_shift = np.array(gaze, dtype=float), None
        self.before = None
        self.landings += 1
        self.at, self.head0, self.far_since = self.landed.copy(), head, None

    def follow(self, head):
        """Between landings the head moves the pointer from where it landed."""
        if self.head0 is None:
            self.head0 = head
        self.at = self.inside(self.landed - self.gain * (head - self.head0))

    def snap_to(self, seq: int, onto) -> bool:
        """Jump `seq` should have landed on `onto` (the button there, `snap.Near`): the
        pointer glides over, the head still moving it from there. Not when another jump
        came since, or the hand has it."""
        if seq != self.landings or self.landed is None or self.at is None:
            return False
        if self.released or self.elsewhere or self.hand_until is not None or self.hand_gaze is not None:
            return False
        moved = self.inside(onto) - self.landed
        self.landed = self.landed + moved
        self.snap_shift = moved if self.snap_shift is None else self.snap_shift + moved
        self.at = self.inside(self.at + moved)
        return True

    def state(self) -> dict:
        """How the pointer got where it is now, as `landing` says it (copies)."""
        copy = lambda v: None if v is None else np.array(v, dtype=float)  # noqa: E731
        return {
            "kind": self.land_kind, "t": self.land_t, "landed": copy(self.landed),
            "gaze": copy(self.land_gaze), "snap": copy(self.snap_shift), "at": copy(self.at),
        }

    def landing(self, t: float) -> dict | None:
        """For a click at `t`, how the pointer got where it is: how it landed (`kind`),
        where (`landed`, after a snap), the gaze then (`gaze`, unclipped: past an edge
        shows), the snap's shift (`snap`), how long ago (`age`, s), what the head did to it
        since (`head`, px) and where it is (`at`). After the hand (or a put), the jump of
        the gaze it took the pointer from, the same way (`before`; its `at` is where the
        hand found the pointer). None when it is not on this monitor."""
        if self.landed is None or self.at is None or self.land_kind is None:
            return None
        out = said(self.state(), t)
        if self.land_kind in ("hand", "put") and self.before is not None:
            out["before"] = said(self.before, t)
        return out

    def inside(self, xy) -> np.ndarray:
        """`xy` kept where the gaze may put the pointer (`area`)."""
        return np.clip(np.asarray(xy, dtype=float), self.area[:2], self.area[2:])

    def head_step(self, head) -> bool:
        """Whether the head is turned to step the strip (`step` set on the frame it has
        stayed there `step_hold` s; until then the head moves the pointer as ever). Back
        from a step, the pointer lands where you look then."""
        left, right = self.step_at
        x = float(head[0])
        if self.turned:
            back = x < left - STEP_BACK if self.turned < 0 else x > right + STEP_BACK
            if not back:
                return True
            self.turned = 0
            self.at = None  # lands on the gaze, with the head as it is now
            self.next_kind = "back"
            return False
        if x >= left or x <= right:
            self.turn_since = self.now if self.turn_since is None else self.turn_since
            if self.now - self.turn_since < self.step_hold:
                return False
            self.step = self.turned = -1 if x >= left else 1
            self.far_since = self.turn_since = None
            return True
        self.turn_since = None
        return False


def said(state: dict, t: float) -> dict:
    """A `Pointer.state` for the clicks' log, at a click at `t`."""
    r1 = lambda v: [round(float(v[0]), 1), round(float(v[1]), 1)]  # noqa: E731
    out = {
        "kind": state["kind"],
        "landed": r1(state["landed"]),
        "age": round(max(t - state["t"], 0.0), 3),
        "head": r1(state["at"] - state["landed"]),
        "at": r1(state["at"]),
    }
    if state["gaze"] is not None:
        out["gaze"] = r1(state["gaze"])
    if state["snap"] is not None:
        out["snap"] = r1(state["snap"])
    return out


def usable(layout: dict | None, size, edge_max: float = EDGE_MAX) -> np.ndarray:
    """Where the gaze may put the pointer on a monitor (x0, y0, x1, y1), from dune's `strip`
    `layout` of it: the tiles on the screen, each edge kept off by what they leave there
    (the outer `gap`, a bar's band), at most `edge_max`; a tile cut by an edge (the strip
    scrolled) leaves nothing there. No layout or no tile on the screen: all of it."""
    w, h = float(size[0]), float(size[1])
    full = np.array([0.0, 0.0, w - 1, h - 1])
    view = (layout or {}).get("view", 0)  # the tiles' x is along the strip: on the screen, less the view
    tiles = [t | {"x": t["x"] - view} for t in (layout or {}).get("tiles", [])]
    tiles = [t for t in tiles if t["x"] < w and t["x"] + t["w"] > 0 and t["y"] < h and t["y"] + t["h"] > 0]
    if not tiles:
        return full
    left = min(max(min(t["x"] for t in tiles), 0.0), edge_max)
    top = min(max(min(t["y"] for t in tiles), 0.0), edge_max)
    right = min(max(w - max(t["x"] + t["w"] for t in tiles), 0.0), edge_max)
    bottom = min(max(h - max(t["y"] + t["h"] for t in tiles), 0.0), edge_max)
    return np.array([left, top, w - 1 - right, h - 1 - bottom])


def unzoom(gaze, zoom: dict | None):
    """Where a gaze on the screen is in real coordinates: inside the area dune magnifies
    (`zoom`, from a `zoom` line) the point seen there, C + (g − C)/scale; elsewhere (or
    not zoomed) itself."""
    if zoom is None:
        return gaze
    g = np.asarray(gaze, dtype=float)
    x0, y0 = zoom["area_x"], zoom["area_y"]
    if not (x0 <= g[0] < x0 + zoom["area_w"] and y0 <= g[1] < y0 + zoom["area_h"]):
        return g
    c = np.array([zoom["x"], zoom["y"]], dtype=float)
    return c + (g - c) / zoom["scale"]


class Filter2:
    """A One Euro filter on each of two numbers."""

    def __init__(self, params: dict):
        self.f = [OneEuroFilter(FPS, **params) for _ in range(2)]

    def __call__(self, value, t: float) -> np.ndarray:
        return np.array([f(float(v), t) for f, v in zip(self.f, value)])


def move(x: int, y: int, output: str | None):
    """`dune pointer move x y output`, straight on dune's socket (27 times a second)."""
    send(f"pointer\tmove\t{x}\t{y}\t{output or ''}")


def action(name: str):
    """`dune action NAME`: what that key of `[keys]` does (`focus_left`...)."""
    send(f"action\t{name}")


def send(request: str):
    path = os.environ.get("DUNE_SOCKET")
    if not path:
        raise RuntimeError("DUNE_SOCKET is not set: not running inside dune?")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(path)
        s.sendall(request.encode())


def run(args) -> int:
    path = geometric.store_path()
    if not path.exists():
        print(f"no calibration at {path}: run `dune-eye-tracker calibrate` first", file=sys.stderr)
        return 1
    try:
        gz = geometric.load(path)
    except Exception as e:
        print(f"the calibration at {path} did not load ({e}): `dune-eye-tracker calibrate` again", file=sys.stderr)
        return 1
    size, output = tuple(gz.screen), gz.output
    try:
        camera = Camera(args.device, share=SHARE, sharing=args.share)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    # `dune-eye-tracker share`: the virtual camera on, or off, for a call or dune-camera
    signal.signal(signal.SIGUSR1, lambda *_: camera.set_sharing(not camera.sharing.is_set()))
    camera.start()
    near = Near(output, args.snap) if args.snap > 0 and output else None
    if near is not None:
        near.start()
    asked = 0  # the last jump `near` was asked about
    around: tuple[int, list] | None = None  # what dune-a11y said is around that jump
    zoom = None  # dune's zoom on this monitor while it is in (a `zoom` line)
    pointer = Pointer(size, head_gain=args.head_gain, head_ref=gz.head_ref(), glide=args.glide, step_at=gz.step_turns())
    gaze_f, head_f = Filter2(GAZE_FILTER), Filter2(HEAD_FILTER)
    learner = geometric.GazeLearner(gz, path)
    clicks = Clicks()
    clicks.start()
    puts = Puts(output) if output else None
    if puts is not None:
        puts.start()
    last_seq, moves, since = 0, 0, time.monotonic()
    shared = None  # what the camera said of sharing, once
    trace = Trace(state_dir() / TRACE_LOG if args.trace else None)  # why the pointer moved (`trace`)
    layout = None  # dune's last `strip` layout of this monitor: where the gaze may put the pointer

    def moment(t: float) -> dict:
        """What a geometric model of clicks needs that the click line does not say: how
        the pointer got there (`jump`)."""
        jump = pointer.landing(t)
        out = {"jump": jump} if jump else {}
        # the clickable things near where the gaze's jump landed: this one, or the one the
        # hand took the pointer from (`before`)
        seq = pointer.landings if jump and jump["kind"] in ("gaze", "back") else None
        if jump and "before" in jump:
            seq = pointer.before_seq
        if seq is not None and around is not None and around[0] == seq:
            out["around"] = around[1]
        return out

    def click(t: float, x: int, y: int, out: str, ctx: dict | None, how: dict) -> str:
        """A click: learnt from (its frames a point, `gaze.GazeLearner`) and logged; the
        shift said when it moves."""
        note = learner.shift_note
        said = learner.click(t, x, y, out, how.get("by"), how | {"hand": pointer.hand_now(t)} | moment(t) | ({"ctx": ctx} if ctx else {}))
        if learner.shift_note != note:
            print(f"gaze: {learner.shift_note}", flush=True)
            trace.event(t, "shift", note=learner.shift_note, offset=[round(float(v), 1) for v in learner.g.offset])
        return said

    clicks_n = len(np.unique(gz.rec.point[gz.rec.block == geometric.CLICK]))
    print(f"driving the pointer on {output} ({size[0]}x{size[1]}); calibrated {gz.time} ({gz.model.name}, "
          f"{len(np.unique(gz.rec.point[gz.rec.block == 0]))} points, {clicks_n} clicks)", flush=True)
    print(f"the clicks keep it fitted: {learner.shift_note}", flush=True)
    if near is not None:
        print(near.error or f"landing on buttons within {args.snap:g} px (dune-a11y, asked at each jump)", flush=True)
    left, right = pointer.step_at
    print(f"a tile left with head_x past {left:+.2f}, right past {right:+.2f}", flush=True)
    try:
        while not stop:
            if camera.error:
                print(camera.error, file=sys.stderr)
                return 1
            if camera.shared != shared:
                shared = camera.shared
                print(shared, flush=True)
                notify(f"eye tracker: {shared}")
            sample = camera.frame()
            if sample.seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = sample.seq
            if (now_layout := clicks.focus.strips.get(output)) is not layout:
                layout = now_layout
                pointer.area = usable(layout, size)
            for _, t, x, y in puts.pending() if puts is not None else ():
                # dune-a11y clicked something there (by voice, "click on X"): a click the
                # least trusted (`clicks.WEIGHTS`), you looked at it somewhere; then the
                # pointer onto it, left there as the mouse leaves it, until you look elsewhere
                said = click(t, x, y, output, None, {"by": "name", "button": "left"})
                if args.verbose:
                    print(f"click on {x} {y} (dune-a11y): {said}", flush=True)
                pointer.hand(t, (x, y), "put")
                trace.event(t, "put", at=[x, y])
                try:
                    move(x, y, output)
                except OSError as e:
                    print(f"cannot reach dune: {e}", file=sys.stderr)
                    return 1
                if args.verbose:
                    print(f"put {x} {y} (dune-a11y)", flush=True)
            for change, t, x, y, out, ctx, how in clicks.pending():
                if change == "zoom":
                    if out == output:
                        # magnified: the gaze mapped back, the head, its wobble and the hand's jump 1/scale
                        zoom = ctx
                        trace.event(t, "zoom", scale=zoom["scale"] if zoom else 1)
                        z = zoom["scale"] if zoom else 1
                        pointer.gain, pointer.dead, pointer.settle, pointer.jump = args.head_gain / z, DEAD / z, SETTLE / z, JUMP / z
                        if args.verbose:
                            print(f"zoom {'in x' + str(z) if zoom else 'out'}", flush=True)
                    continue
                if change == "move":
                    pointer.hand(t, (x, y) if out == output else None)
                    continue
                said = click(t, x, y, out, ctx, how)
                if args.verbose:
                    print(f"click {x} {y} {out}: {said}", flush=True)
            if note := learner.refit_background():  # fits in a thread: the pointer goes on
                print(f"gaze: {note}", flush=True)
                trace.event(sample.t, "fit", note=note)
            feats = sample.feats
            if feats is None:
                continue
            learner.seen(sample)
            guess = gz.predict([sample]) if feats["open"] >= BLINK else None  # unclipped: off the screen is how looking away shows
            if guess is None:  # blinking, or no iris found this frame: the pointer stays
                trace.frame(sample.t, None, None, (feats["head_x"], feats["head_y"]), pointer.shown, pointer.released, feats["open"])
                continue
            gaze = gaze_f(guess[0], sample.t)
            head = head_f((feats["head_x"], feats["head_y"]), sample.t)
            was_released = pointer.released
            if near is not None:
                for seq, onto, found in near.answers():  # where a jump should have landed
                    if found is not None:
                        around = (seq, found)
                    if onto is not None and pointer.snap_to(seq, onto):
                        trace.event(sample.t, "snap", jump=seq, onto=[round(float(onto[0])), round(float(onto[1]))])
                        if args.verbose:
                            print(f"jump {seq} onto {onto[0]:.0f} {onto[1]:.0f}", flush=True)
            landings = pointer.landings
            px = pointer.update(sample.t, unzoom(gaze, zoom), head)
            trace.frame(sample.t, guess[0], gaze, head, pointer.shown, pointer.released, feats["open"])
            if pointer.landings != landings and trace.jump(sample.t, pointer.land_kind, pointer.came_from, pointer.landed, guess[0]):
                j = trace.lines[-1]
                print(f"bounce: to {j['went']} and back to {j['to']} after {j['back_after']} s (trace.jsonl)", flush=True)
            if near is not None and zoom is None and pointer.landings != asked and pointer.landed is not None:
                asked = pointer.landings  # a new jump: what is there? (answered in a few ms)
                near.ask(asked, *pointer.landed)
            if pointer.released != was_released:
                print("looking off the screen: pointer let go" if pointer.released else "back on the screen", flush=True)
            if pointer.step and args.verbose:
                side = "left" if pointer.step < 0 else "right"
                print(f"head turned {side} (head_x {head[0]:+.2f}): one tile {side}", flush=True)
            if px is not None or pointer.step:
                try:
                    if px is not None:
                        move(*px, output)
                    if pointer.step:
                        action("focus_left" if pointer.step < 0 else "focus_right")
                except OSError as e:  # dune went away (restarted, quit)
                    print(f"cannot reach dune: {e}", file=sys.stderr)
                    return 1
                moves += 1
            if args.verbose and sample.t - since >= 1.0:
                print(f"{camera.fps:4.1f} fps  {moves} moves  pointer {pointer.sent}  head_x {head[0]:+.2f}", flush=True)
                moves, since = 0, sample.t
    finally:
        clicks.stop()
        if puts is not None:
            puts.stop()
        if near is not None:
            near.stop()
        if learner.kept:
            learner.save()
        camera.stop()
        camera.join(timeout=2)  # out of MediaPipe before Python goes (else it aborts)
        if camera.is_alive():  # OpenCV's release of the camera can take seconds: the
            os._exit(0)  # kernel closes it as well, without Python's teardown
    return 0


def pid_path() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "dune-eye-tracker.pid"


def running() -> int | None:
    """The pid of a `run` going (the one `toggle` started, else any other, found in
    /proc: a lost pid file must not leave one that cannot be stopped)."""
    try:
        pid = int(pid_path().read_text())
        os.kill(pid, 0)
        if is_run(pid):
            return pid
    except (OSError, ValueError):
        pass
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) != os.getpid() and is_run(int(entry.name)):
            return int(entry.name)
    return None


def is_run(pid: int) -> bool:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return b"dune_eye_tracker.cli" in argv and b"run" in argv


def notify(text: str):
    try:
        subprocess.run(["notify-send", "-a", "dune-eye-tracker", "-t", "1500", text], check=False, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        pass


def share(args) -> int:
    """Turn the virtual camera on, or off, in the `run` going (SIGUSR1): other programs
    read the webcam there while the pointer goes on."""
    pid = running()
    if not pid:
        print("the tracker is not running: the webcam is free, no need to share it", file=sys.stderr)
        notify("eye tracker: not running, the webcam is free")
        return 1
    os.kill(pid, signal.SIGUSR1)
    return 0


def toggle(args) -> int:
    """Stop the `run` going, else start one in the background (with no calibration yet:
    `calibrate` first)."""
    pid = running()
    if pid:
        while pid:  # every one going (two could, started by hand)
            os.kill(pid, signal.SIGTERM)
            for _ in range(50):  # gone within 5 s (the camera's release can take 2)
                if not is_run(pid):
                    break
                time.sleep(0.1)
            else:
                os.kill(pid, signal.SIGKILL)
            pid = running()
        pid_path().unlink(missing_ok=True)
        notify("eye tracker: off")
        return 0
    command = "run" if geometric.store_path().exists() else "calibrate"
    log = state_dir() / f"{command}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        child = subprocess.Popen(
            [sys.executable, "-m", "dune_eye_tracker.cli", command],
            stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True,
        )
    if command == "calibrate":
        return 0
    pid_path().write_text(str(child.pid))
    try:  # the camera busy (dune-camera), no face model...: it says so within a second
        child.wait(timeout=2.5)
    except subprocess.TimeoutExpired:
        notify("eye tracker: on")
        return 0
    pid_path().unlink(missing_ok=True)
    notify(f"eye tracker: could not start, see {log}")
    return 1
