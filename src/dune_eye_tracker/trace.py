"""The pointer's trace: why it moved, to find what makes it jump there and back.

`run` keeps the last `KEEP` frames in memory (the gaze as guessed, filtered, the head, the
pointer, its speed) and writes to `trace.jsonl` beside the calibration:

- `jump`: each jump of the pointer (how: the gaze, back from a head step, the hand, a put),
  from where to where, how far, how fast it glides (px/s) and how long since the last one;
- `bounce`: a jump that goes back near where the pointer was before the last one (within
  `BACK` px, `BOUNCE_TIME` s after it, the last one `FAR` px or more away): what made the
  pointer go there and back, with the frames of the last `KEEP` (~3 s);
- `event`: what else moves it or the gaze under it (a snap onto a button, the clicks'
  shift, a new fit, the zoom), so a bounce can be put next to it.

Numbers only, as the rest; `dune-eye-tracker trace` sums it up.
"""

from __future__ import annotations

import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path

LOG = "trace.jsonl"
KEEP = 90  # frames kept in memory (~3 s): what a bounce is written with
BACK = 150.0  # px: a jump that lands this near where the pointer was before the last one
BOUNCE_TIME = 1.5  # s after the last jump, at most
FAR = 300.0  # px: the last jump at least this long (a real jump, not the head's last pixels)


def _r(v) -> list[float] | None:
    return None if v is None else [round(float(v[0]), 1), round(float(v[1]), 1)]


class Trace:
    def __init__(self, path: Path | None):
        self.path = path  # None: kept in memory only (tests)
        self.frames: deque = deque(maxlen=KEEP)
        self.jumps: deque = deque(maxlen=3)  # (t, from, to, kind) of the last jumps
        self.last_at = None  # the pointer at the last frame, and when
        self.last_t = 0.0
        self.lines: list[dict] = []  # what was written (also kept, for tests and the summary)
        self.bounces = 0

    def write(self, line: dict):
        line = {"time": datetime.now().isoformat(timespec="milliseconds")} | line
        self.lines.append(line)
        if len(self.lines) > 1000:
            del self.lines[:500]
        if self.path is None:
            return
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:  # a full disk must not stop the pointer
            pass

    def frame(self, t: float, raw, gaze, head, at, released: bool, opened: float):
        """One frame: the gaze guessed (`raw`, the clicks' shift taken off, not filtered),
        filtered (`gaze`), the head, where the pointer is shown (`at`, None: not here)."""
        speed = None
        if at is not None and self.last_at is not None and t > self.last_t:
            speed = math.hypot(at[0] - self.last_at[0], at[1] - self.last_at[1]) / (t - self.last_t)
        self.frames.append({
            "t": round(t, 3), "raw": _r(raw), "gaze": _r(gaze), "head": [round(float(head[0]), 3), round(float(head[1]), 3)],
            "at": _r(at), "speed": None if speed is None else round(speed), "released": released, "open": round(float(opened), 3),
        })
        self.last_at, self.last_t = (None if at is None else (float(at[0]), float(at[1]))), t

    def jump(self, t: float, kind: str | None, start, end, gaze=None) -> bool:
        """The pointer jumped (`kind`) from `start` (None: it was not here) to `end`; true
        when it is a bounce (written with the frames)."""
        far = None if start is None else math.hypot(end[0] - start[0], end[1] - start[1])
        since = None if not self.jumps else t - self.jumps[-1][0]
        line = {"kind": "jump", "how": kind, "from": _r(start), "to": _r(end), "far": None if far is None else round(far),
                "since": None if since is None else round(since, 3), "gaze": _r(gaze)}
        if far is not None:
            line["speed"] = round(far / 0.1)  # it glides most of the way in `GLIDE` (0.1 s)
        bounce = False
        if self.jumps and start is not None:
            t0, a, b, _ = self.jumps[-1]
            if (a is not None and t - t0 <= BOUNCE_TIME and math.hypot(b[0] - a[0], b[1] - a[1]) >= FAR
                    and math.hypot(end[0] - a[0], end[1] - a[1]) <= BACK):
                bounce = True
                self.bounces += 1
                line |= {"kind": "bounce", "went": _r(b), "back_after": round(t - t0, 3), "frames": list(self.frames)}
        self.write(line)
        self.jumps.append((t, None if start is None else (float(start[0]), float(start[1])), (float(end[0]), float(end[1])), kind))
        return bounce

    def event(self, t: float, what: str, **more):
        """Something else that moves the pointer or the gaze: `snap`, `shift`, `fit`, `zoom`, `hand`."""
        self.write({"kind": "event", "what": what, "t": round(t, 3)} | more)


def summary(lines: list[dict]) -> list[str]:
    """`dune-eye-tracker trace`: jumps, bounces, and what came before each bounce."""
    jumps = [x for x in lines if x.get("kind") in ("jump", "bounce")]
    bounces = [x for x in lines if x.get("kind") == "bounce"]
    if not jumps:
        return ["no jumps traced yet: `run` writes them (trace.jsonl)"]
    kinds: dict[str, int] = {}
    for j in jumps:
        kinds[j.get("how") or "?"] = kinds.get(j.get("how") or "?", 0) + 1
    out = [f"{len(jumps)} jumps ({', '.join(f'{k} {n}' for k, n in sorted(kinds.items()))}), {len(bounces)} bounces"]
    for b in bounces[-10:]:
        fr = b.get("frames") or []
        raws = [f["raw"] for f in fr if f.get("raw")]
        spread = 0.0
        if len(raws) > 2:
            mx, my = sum(r[0] for r in raws) / len(raws), sum(r[1] for r in raws) / len(raws)
            spread = math.sqrt(sum((r[0] - mx) ** 2 + (r[1] - my) ** 2 for r in raws) / len(raws))
        shut = sum(1 for f in fr if f.get("open", 1) < 0.15)
        out.append(f"  {b['time']}: {b.get('how')} to {b.get('went')} and back to {b.get('to')} after {b.get('back_after')} s;"
                   f" the raw gaze spread {spread:.0f} px over {len(fr)} frames, {shut} nearly shut")
    return out
