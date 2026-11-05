"""`dune-eye-tracker game`: how far off the tracker is, then how well it plays.

A window fullscreen on the calibrated monitor. Two rounds (`--only` one of them):

- measure: the tracker itself. One point at a time on a fixed grid (`POINTS`: the edges
  in, none a calibration point, the same every game so games can be held against each
  other), no pointer shown (seeing it, you would look off to put it right, and that
  would be measured instead). Look at each until it is gone; the gaze is taken `WINDOW`
  s after it shows (the eyes there, steady). The head may move as in use.
- play: one or two circles at a time (`AT_ONCE`), of radius 30 to 180 px (`RADII`), big
  and small in turn (`order`), spread evenly over the quadrants (`place`). No click: the
  pointer kept inside a circle `DWELL` s takes it; after `TIMEOUT` s it is missed. The
  pointer moves as `run` moves it (the same gaze, its clicks' shift, filters and
  `track.Pointer`), without the mouse, dune-a11y's snap or the head's tile steps. What is
  taken and how fast: whether it can be used. Its misses are not the tracker's (a circle
  taken was reached, whatever the gaze; with two up, a frame counts for the nearer one).

`run` has to be off: the game needs the camera. Each point and circle is a line in
`game.jsonl` beside `clicks.jsonl`, shaped like a click's (`seen`, `guess`, `spread`,
`feats`). The measure round is also saved as a recording, `games/<game>.npz`, with the
calibration it was played with (its `grid` block the points): `dune-eye-tracker eval
games/*.npz` guesses it again with every model. The end (and `game --report`, all the
games of the current gaze) gives the medians with a 95% interval (`ci`), per quadrant,
against the click log and the goal (`GOAL`).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import gaze as geometric
from . import track
from .camera import BLINK, Camera
from .clicks import AFTER, BEFORE, LOG, MIN_SAMPLES
from .record import BACKGROUND, DIM, TARGET, TEXT, WARN
from .recording import BLOCKS, build, state_dir

# measure: a 6 x 4 grid, 6 points a quadrant, the edges in, none where `calibrate` looks
POINTS = [(x, y) for y in (0.06, 0.3, 0.7, 0.94) for x in (0.04, 0.17, 0.39, 0.61, 0.83, 0.96)]
LOOK = 1.8  # s a point is shown
WINDOW = (0.6, 1.5)  # s after it shows: the frames its gaze is taken from

# play
RADII = (30, 50, 75, 100, 120, 150, 180)  # px
ROUNDS = 4  # circles of each radius in a game: 28, 7 a quadrant
AT_ONCE = (1, 2)  # circles up at a time: one of these, drawn again after each circle
DWELL = 0.5  # s the pointer stays inside: taken
TIMEOUT = 6.0  # s a circle waits: missed
PAUSE = 0.7  # s before a new circle shows
GAP = 150.0  # px at least between the edges of two circles up together
EDGE = 20.0  # px between a circle and the screen's edge
SETTLED = 0.6  # s after a circle shows from which the gaze is taken as on it

APART = 350.0  # px at least from the last centre: more than `track.DEAD`, the gaze moves it
BREAK = 2.5  # s between the rounds
GOAL = (100.0, 120.0)  # px: the median miss we want
GAME_LOG = "game.jsonl"
QUADRANTS = ("top left", "top right", "bottom left", "bottom right")

HIT = (74, 222, 128)
GAZE = (251, 191, 36)


# --- where things go ---------------------------------------------------------------


def quadrant(x: float, y: float, size) -> int:
    """0 top left, 1 top right, 2 bottom left, 3 bottom right."""
    return (2 if y >= size[1] / 2 else 0) + (1 if x >= size[0] / 2 else 0)


def point_order(rng: np.random.Generator, size, points=POINTS, apart: float = APART) -> list[int]:
    """The grid's points in a random order, each `apart` from the last when one is left
    that is (the gaze jumps, as in use)."""
    w, h = size
    left = list(range(len(points)))
    rng.shuffle(left)
    out: list[int] = []
    while left:
        far = [i for i in left if not out or math.hypot((points[i][0] - points[out[-1]][0]) * w, (points[i][1] - points[out[-1]][1]) * h) >= apart]
        i = (far or left)[0]
        left.remove(i)
        out.append(i)
    return out


def order(rng: np.random.Generator, radii=RADII, rounds: int = ROUNDS) -> list[float]:
    """The radii of a game, going big, small, big...: each round the bigger half and the
    smaller half shuffled, then taken in turn (an odd one in the middle goes last)."""
    rs = sorted(float(r) for r in radii)
    half = len(rs) // 2
    out: list[float] = []
    for _ in range(rounds):
        small, big = rs[:half], rs[len(rs) - half :]
        rng.shuffle(small)
        rng.shuffle(big)
        for b, s in zip(big, small):
            out += [b, s]
        out += rs[half : len(rs) - half]
    return out


def place(size, r: float, rng: np.random.Generator, counts: list[int], up=(), last=None, apart: float = APART) -> tuple[float, float, int]:
    """Where a circle of radius `r` goes, and its quadrant: the quadrant with the fewest
    circles so far (`counts`; not one a circle is up in, when there is another), whole on the
    screen, `apart` from the `last` centre and `GAP` from the circles `up`."""
    w, h = size
    busy = {quadrant(x, y, size) for x, y, _ in up}
    free = [q for q in range(4) if q not in busy] or list(range(4))
    fewest = min(counts[q] for q in free)
    q = int(rng.choice([q for q in free if counts[q] == fewest]))
    x0, y0 = (w / 2 if q % 2 else 0.0), (h / 2 if q >= 2 else 0.0)
    lo = (max(x0, r + EDGE), max(y0, r + EDGE))
    hi = (min(x0 + w / 2, w - r - EDGE), min(y0 + h / 2, h - r - EDGE))
    x = y = 0.0
    for _ in range(300):
        x, y = float(rng.uniform(lo[0], hi[0])), float(rng.uniform(lo[1], hi[1]))
        if last is not None and math.hypot(x - last[0], y - last[1]) < apart:
            continue
        if any(math.hypot(x - ux, y - uy) < r + ur + GAP for ux, uy, ur in up):
            continue
        break
    return x, y, q


def nearest(guess, up) -> int:
    """The circle up whose centre the gaze is nearest (an index of `up`)."""
    return min(range(len(up)), key=lambda i: math.hypot(guess[0] - up[i][0], guess[1] - up[i][1]))


# --- a point, a circle ------------------------------------------------------------


def _median_of(rows: list[dict], key: str) -> np.ndarray:
    return np.median(np.array([f[key] for f in rows], dtype=float), axis=0)


def _r1(v) -> list[float]:
    return [round(float(v[0]), 1), round(float(v[1]), 1)]


def point_line(frames: list[dict], target, shown: float) -> dict:
    """A point of the measure round, for `game.jsonl`. `frames`: the eyes-open frames while
    it showed, each `t` (s), `guess` (the gaze as `run` has it, the clicks' shift taken
    off, not filtered), `raw` (without the shift) and `feats`. From those `WINDOW` s after
    it showed: `guess` (the median), `spread`, `feats`, `miss` (from `seen`) and `raw_miss`
    (without the shift)."""
    x, y = target
    centre = np.array([x, y], dtype=float)
    line = {"seen": [round(x, 1), round(y, 1)]}
    rows = [f for f in frames if shown + WINDOW[0] <= f["t"] <= shown + WINDOW[1]]
    if len(rows) >= MIN_SAMPLES:
        mid = _median_of(rows, "guess")
        g = np.array([f["guess"] for f in rows], dtype=float)
        line["guess"] = _r1(mid)
        line["spread"] = round(float(np.sqrt((np.linalg.norm(g - mid, axis=1) ** 2).mean())), 1)
        line["miss"] = round(float(np.linalg.norm(mid - centre)), 1)
        line["raw_miss"] = round(float(np.linalg.norm(_median_of(rows, "raw") - centre)), 1)
        line["feats"] = {k: float(np.median([f["feats"][k] for f in rows])) for k in rows[0]["feats"]}
    return line


def trial_line(frames: list[dict], target, shown: float, end: float, hit: bool, jump: dict | None = None, at=None) -> dict:
    """A circle of the play round, for `game.jsonl`. `frames`: the eyes-open frames that
    counted for it, each `t` (s), `guess` (as `run` has it, not filtered) and `feats`. Like a
    click's line: `seen` (the centre), `guess` (the median of the frames `BEFORE`..`AFTER` s
    before the end), `spread`, `feats`; and `radius`, `hit`, `took` (s), `miss`, `settled`
    (the median guess from `SETTLED` s on, from the centre) and `pointer` (`at`, where the
    pointer was at the end, from the centre)."""
    x, y, r = target
    centre = np.array([x, y], dtype=float)
    line = {"seen": [round(x, 1), round(y, 1)], "radius": r, "hit": hit, "took": round(end - shown, 3)}
    rows = [f for f in frames if end - BEFORE <= f["t"] <= end - AFTER]
    if len(rows) >= MIN_SAMPLES:
        g = np.array([f["guess"] for f in rows], dtype=float)
        mid = np.median(g, axis=0)
        line["guess"] = _r1(mid)
        line["spread"] = round(float(np.sqrt((np.linalg.norm(g - mid, axis=1) ** 2).mean())), 1)
        line["miss"] = round(float(np.linalg.norm(mid - centre)), 1)
        line["feats"] = {k: float(np.median([f["feats"][k] for f in rows])) for k in rows[0]["feats"]}
    late = [f["guess"] for f in frames if f["t"] >= shown + SETTLED]
    if len(late) >= MIN_SAMPLES:
        line["settled"] = round(float(np.linalg.norm(np.median(np.array(late, dtype=float), axis=0) - centre)), 1)
    if at is not None:
        line["pointer"] = round(float(np.linalg.norm(np.asarray(at, dtype=float) - centre)), 1)
    if jump:
        line["jump"] = jump
    return line


def conditions(feats: list[dict], fps: list[float], hours: float | None) -> dict:
    """How a game was played: `distance` (the median `pos_z`, cm from the camera), how
    much the head `moved` (cm) and `turned` (head axis units; 0.05 is about 3°: the
    spread of each, both axes), the median `fps`, and `hours` since the calibration."""
    out: dict = {}
    if feats:
        def sd(*keys):
            return round(float(np.sqrt(sum(np.var([f[k] for f in feats]) for k in keys))), 3)

        out = {"distance": round(abs(float(np.median([f["pos_z"] for f in feats]))), 1), "moved": sd("pos_x", "pos_y"), "turned": sd("head_x", "head_y")}
    if fps:
        out["fps"] = round(float(np.median(fps)), 1)
    if hours is not None:
        out["hours"] = round(hours, 1)
    return out


# --- numbers ------------------------------------------------------------------------


def median(xs) -> float | None:
    return float(np.median(xs)) if len(xs) else None


def ci(xs, n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """A 95% bootstrap interval of the median of `xs` (None: fewer than 3)."""
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 3:
        return None
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(xs, size=(n, len(xs)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def med_ci(xs) -> str:
    """'142 (120-165)': the median and its interval, px."""
    m = median(xs)
    if m is None:
        return "-"
    c = ci(xs)
    return f"{m:.0f}" + ("" if c is None else f" ({c[0]:.0f}-{c[1]:.0f})")


def goal_said(xs) -> str:
    """Whether the median is within the goal, as far as its interval tells."""
    lo, hi = GOAL
    m, c = median(xs), ci(xs)
    if m is None:
        return "-"
    if c is None:
        return f"{m:.0f} px, too few to tell"
    if c[1] <= hi:
        return "reached"
    if c[0] > hi:
        return f"not yet ({m - hi:.0f} px to go)"
    return f"cannot tell yet (the interval spans {hi:.0f}): more games"


def read_lines(path: Path) -> list[dict]:
    try:
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    except (OSError, ValueError):
        return []


def click_misses(clicks: list[dict]) -> list[float]:
    """How far the guess was from each click of the click log (px)."""
    return [math.hypot(c["guess"][0] - c["seen"][0], c["guess"][1] - c["seen"][1]) for c in clicks if c.get("guess") and c.get("seen")]


def by_quadrant(lines: list[dict], clicks: list[dict]) -> list[str]:
    """Per quadrant of where the target was: how many, the median miss (95% interval) and
    how the guess leans (dx, dy: + right / down); the miss of the targets whose guess fell
    there (all the tracker knows in use); the click log's miss there."""
    out = [f"{'quadrant':>13} {'n':>4} {'miss (95%)':>15} {'lean':>11} {'by guess':>9} {'clicks':>7}"]
    for q, name in enumerate(QUADRANTS):
        these = [g for g in lines if "guess" in g and quadrant(*g["seen"], g["size"]) == q]
        dx = median([g["guess"][0] - g["seen"][0] for g in these])
        dy = median([g["guess"][1] - g["seen"][1] for g in these])
        seen_q = median([g["miss"] for g in lines if "guess" in g and quadrant(*g["guess"], g["size"]) == q])
        cm = click_misses([c for c in clicks if c.get("seen") and c.get("size") and quadrant(*c["seen"], c["size"]) == q])
        lean = "" if dx is None else f"{dx:+.0f},{dy:+.0f}"
        out.append(
            f"{name:>13} {len(these):>4} {med_ci([g['miss'] for g in these]):>15} {lean:>11}"
            f" {'' if seen_q is None else f'{seen_q:.0f}':>9} {'' if not cm else f'{median(cm):.0f}':>7}"
        )
    return out


def report(lines: list[dict], clicks: list[dict], title: str = "this game") -> list[str]:
    """The measure round (the tracker: the miss with its interval, per quadrant, the goal),
    the play round (what was taken per radius), the conditions, and the click log."""
    if not lines:
        return ["no game yet: play one (dune-eye-tracker game)"]
    points = [g for g in lines if g.get("round") == "measure"]
    circles = [g for g in lines if g.get("round", "play") == "play"]
    out = [f"{title}: {len(points)} points measured, {len(circles)} circles played", ""]
    if points:
        seen = [g for g in points if "miss" in g]
        raw = [g.get("raw_miss", g.get("eyes_miss")) for g in seen]
        out.append(f"the tracker (points, no pointer shown): miss {med_ci([g['miss'] for g in seen])} px, without the clicks' shift {med_ci([r for r in raw if r is not None])} px")
        out.append(f"goal {GOAL[0]:.0f}-{GOAL[1]:.0f} px: {goal_said([g['miss'] for g in seen])}")
        out += [""] + by_quadrant(seen, clicks) + [""]
    if circles:
        hits = sum(1 for g in circles if g["hit"])
        out.append(f"play: {hits}/{len(circles)} circles taken ({100 * hits / len(circles):.0f}%)")
        out.append(f"{'radius':>7} {'taken':>7} {'took':>7}")
        for r in sorted({g["radius"] for g in circles}):
            these = [g for g in circles if g["radius"] == r]
            n = sum(1 for g in these if g["hit"])
            took = median([g["took"] for g in these if g["hit"]])
            out.append(f"{r:>7.0f} {f'{n}/{len(these)}':>7} {'' if took is None else f'{took:.1f} s':>7}")
        out.append("")
    conds = [g["cond"] for g in points + circles if g.get("cond")]
    if conds:
        def m(k):
            v = median([c[k] for c in conds if k in c])
            return "-" if v is None else f"{v:g}"

        out.append(f"conditions: {m('distance')} cm away, head moved {m('moved')} cm and turned {m('turned')}, {m('fps')} fps, {m('hours')} h after calibrating")
    cm = click_misses(clicks)
    if cm:
        out.append(f"the click log: {med_ci(cm)} px over {len(cm)} clicks, the last 50: {med_ci(cm[-50:])} px")
    return out


# --- files --------------------------------------------------------------------------


def game_paths() -> tuple[Path, Path, Path]:
    """`game.jsonl`, `clicks.jsonl` and the folder of the games' frames."""
    d = state_dir()
    return d / GAME_LOG, d / LOG, d / "games"


def report_file() -> int:
    """`game --report`: the games of the current gaze (calibration, clicks, fit), else all."""
    log, clicks, _ = game_paths()
    lines = read_lines(log)
    try:
        g = geometric.load()
    except Exception:
        g = None
    mid = g.model_id() if g else None
    these = [x for x in lines if x.get("model") == mid]
    if these:
        n = len({x["game"] for x in these})
        title = f"{n} game{'s' if n > 1 else ''} with the current gaze ({mid})"
    else:
        these, title = lines, "all games (none with the current gaze)"
    print("\n".join(report(these, read_lines(clicks), title)))
    return 0


# --- the window ---------------------------------------------------------------------


def run(args) -> int:
    if args.report:
        return report_file()
    path = geometric.store_path()
    if not path.exists():
        print(f"no calibration at {path}: run `dune-eye-tracker calibrate` first", file=sys.stderr)
        return 1
    if track.running():
        print("the tracker is running and has the camera: stop it first (dune-eye-tracker toggle)", file=sys.stderr)
        return 1
    gz = geometric.load(path)
    geometric.GazeLearner(gz, path)  # the clicks' shift, as `run` has it at its start
    size, output = tuple(gz.screen), gz.output
    mid = gz.model_id()
    log_path, clicks_path, games_dir = game_paths()
    try:
        hours0 = (datetime.now() - datetime.fromisoformat(gz.time)).total_seconds() / 3600
    except (TypeError, ValueError):
        hours0 = None
    rounds = ["measure", "play"] if args.only is None else [args.only]

    os.environ.setdefault("SDL_VIDEODRIVER", "wayland,x11")
    os.environ.setdefault("SDL_VIDEO_WAYLAND_WMCLASS", "dune-eye-tracker")
    import pygame

    try:
        camera = Camera(args.device)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    camera.start()

    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("dune-eye-tracker game")
    font = pygame.font.SysFont("JetBrains Mono,DejaVu Sans Mono,monospace", 22)
    big = pygame.font.SysFont("JetBrains Mono,DejaVu Sans Mono,monospace", 30)
    clock = pygame.time.Clock()
    asked_fullscreen = not args.fullscreen or shutil.which("dune") is None
    total = len(RADII) * args.rounds
    minutes = (("measure" in rounds) * len(POINTS) * LOOK + ("play" in rounds) * total * (TIMEOUT + PAUSE)) / 60

    rng = np.random.default_rng()
    phase = "wait"  # wait, measure, break, play, done
    g: dict = {}  # the game going: everything below is reset at Space
    summary: list[str] = []
    last_seq = 0
    note = ""

    def text(rows, y, f=font, colour=TEXT):
        for row in rows:
            surf = f.render(row, True, colour)
            screen.blit(surf, ((screen.get_width() - surf.get_width()) // 2, y))
            y += surf.get_height() + 8

    def write_lines(lines: list[dict]):
        try:
            with open(log_path, "a") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
        except OSError:
            pass

    def head_of(round_name: str) -> dict:
        return {"time": datetime.now().isoformat(timespec="milliseconds"), "game": g["id"], "round": round_name, "calibration": gz.time, "model": mid, "output": output, "size": list(size)}

    def cond() -> dict:
        hours = None if hours0 is None else hours0 + (time.monotonic() - g["t0"]) / 3600
        return conditions(g["feats"][-3000:], g["fps"], hours)

    def end_measure():
        """The measure round's lines, with the conditions so far, once it is over."""
        c = cond()
        lines = [head_of("measure") | {"point": p["point"], "quadrant": quadrant(*p["target"], size)} | point_line(p["frames"], p["target"], p["shown"]) | {"cond": c} for p in g["measure"]]
        g["lines"] += lines
        write_lines(lines)
        g["measured"] = True

    def save_game():
        """The measure round as a recording (the calibration with it: `eval` replays it), the
        play round's circles (their frames' guesses, no landmarks) as JSON."""
        if not g or (not g["measure"] and not g["play"]):
            return
        games_dir.mkdir(parents=True, exist_ok=True)
        name = g["id"].replace(":", "-")
        rows = [
            {"landmarks": f["frame"].landmarks, "matrix": f["frame"].matrix, "t": f["t"], "block": BLOCKS.index("grid"), "point": n,
             "target": p["target"], "since": f["t"] - p["shown"], "image": f["frame"].image,
             "limbus": f["frame"].eyes["limbus"], "dark": f["frame"].eyes["dark"], "maps": f["frame"].eyes["maps"]}
            for n, p in enumerate(g["measure"]) for f in p["frames"]
        ]
        if rows:
            meta = {"screen": list(size), "image": list(rows[0]["image"]), "time": g["id"], "game": True, "model": mid, "output": output}
            cal = geometric.subset(gz.rec, gz.rec.block == BLOCKS.index("calibration"))
            geometric.concat(cal, build(rows, meta)).save(games_dir / f"{name}.npz")
        strip = lambda frames: [{k: f[k] for k in ("t", "guess", "feats")} for f in frames]  # noqa: E731
        out = {
            "game": g["id"], "calibration": gz.time, "model_id": mid, "output": output, "size": list(size), "conditions": cond(),
            "measure": [{k: p[k] for k in ("point", "target", "shown")} for p in g["measure"]],
            "play": [p | {"frames": strip(p["frames"])} for p in g["play"]],
        }
        (games_dir / f"{name}.json").write_text(json.dumps(out))

    def start(now: float):
        nonlocal phase
        g.clear()
        g.update(
            id=time.strftime("%Y-%m-%dT%H:%M:%S"), t0=now, lines=[], feats=[], fps=[], measure=[], play=[], measured=False,
            order=point_order(rng, size), point=None, queue=order(rng, rounds=args.rounds), up=[], counts=[0, 0, 0, 0],
            want=1, next_at=now + PAUSE, last=None, flash=None, pointer=None,
        )
        phase = rounds[0]
        if phase == "play":
            begin_play(now)
        else:
            next_point(now)
        pygame.mouse.set_visible(False)

    def next_point(now: float):
        if g["point"] is not None:
            g["measure"].append(g["point"])
        if not g["order"]:
            g["point"] = None
            return
        i = g["order"].pop(0)
        x, y = POINTS[i]
        g["point"] = {"point": i, "target": [x * size[0], y * size[1]], "shown": now, "frames": []}

    def begin_play(now: float):
        g["pointer"] = (
            track.Pointer(size, head_gain=args.head_gain, head_ref=gz.head_ref(), glide=args.glide, step_at=(math.inf, -math.inf)),
            track.Filter2(track.GAZE_FILTER),
            track.Filter2(track.HEAD_FILTER),
        )
        g["next_at"] = now + PAUSE

    def pointer_at():
        p = g["pointer"][0]
        return None if p.released or p.shown is None else p.shown.copy()

    def finish(c: dict, t: float, taken: bool):
        p = g["pointer"][0]
        others = [[round(o["x"], 1), round(o["y"], 1), o["r"]] for o in g["up"] if o is not c]
        line = head_of("play") | trial_line(c["frames"], (c["x"], c["y"], c["r"]), c["shown"], t, taken, p.landing(t), pointer_at())
        line |= {"quadrant": c["q"], "with": others, "cond": cond()}
        g["lines"].append(line)
        g["play"].append({"target": [c["x"], c["y"], c["r"]], "quadrant": c["q"], "shown": c["shown"], "end": t, "hit": taken, "with": others, "frames": c["frames"]})
        write_lines([line])
        g["up"].remove(c)
        g["last"], g["flash"] = (c["x"], c["y"]), (taken, t + PAUSE, c["x"], c["y"])
        g["want"], g["next_at"] = int(rng.choice(AT_ONCE)), t + PAUSE

    def spawn(now: float):
        r = g["queue"].pop(0)
        x, y, q = place(size, r, rng, g["counts"], [(c["x"], c["y"], c["r"]) for c in g["up"]], g["last"])
        g["counts"][q] += 1
        g["up"].append({"x": x, "y": y, "r": r, "q": q, "shown": now, "inside": None, "frames": []})

    def done():
        nonlocal phase, summary
        phase = "done"
        save_game()
        summary = report(g["lines"], read_lines(clicks_path))
        print("\n".join(summary), flush=True)
        pygame.mouse.set_visible(True)

    fps_since = 0.0
    try:
        while True:
            now = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return 0
                if event.type == pygame.WINDOWFOCUSGAINED and not asked_fullscreen:
                    asked_fullscreen = True  # dune ignores it before the window is a tile
                    subprocess.run(["dune", "action", "fullscreen"], check=False)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and phase in ("wait", "done"):
                    w, h = screen.get_size()
                    if (w, h) != size:
                        note = f"this window is {w}x{h}, the calibration {size[0]}x{size[1]} on {output}: fullscreen there"
                    elif camera.frame().feats is None:
                        note = "no face yet: look at the screen, light on your face"
                    else:
                        note = ""
                        start(now)

            w, h = screen.get_size()
            screen.fill(BACKGROUND)
            if camera.error:
                text([camera.error, "Esc: quit"], h // 2 - 30)
                pygame.display.flip()
                clock.tick(30)
                continue
            sample = camera.frame()
            status = f"{camera.fps:4.1f} fps   face: {'yes' if sample.feats else 'NO'}"
            fresh = sample.seq != last_seq and sample.feats is not None
            if fresh:
                last_seq = sample.seq
            if phase in ("measure", "break", "play") and now - fps_since >= 1.0:
                g["fps"].append(camera.fps)
                fps_since = now
            feats = sample.feats if fresh and sample.feats["open"] >= BLINK else None
            if feats is not None and phase in ("measure", "break", "play"):
                g["feats"].append(feats)

            if phase == "wait":
                text(["dune-eye-tracker: the circles game", ""], h // 3 - 40, big)
                text(
                    [
                        f"First, {len(POINTS)} points, no pointer shown: look at each until it is gone. This measures the tracker." if "measure" in rounds else "",
                        "Then circles, one or two at a time: bring the yellow pointer into one with your eyes" if "play" in rounds else "",
                        f"and keep it there {DWELL:g} s (no click; the head moves it the last pixels, the mouse does nothing)." if "play" in rounds else "",
                        f"{total} circles, big and small in turn, each waits {TIMEOUT:g} s." if "play" in rounds else "",
                        f"Sit as you usually do. At most {minutes:.0f} min.",
                        "",
                        "Space: start     Esc: quit",
                    ],
                    h // 3 + 30,
                )
                if note:
                    text([note], h // 3 + 300, colour=WARN)
                text([status], h - 60, colour=DIM)

            elif phase == "measure":
                p = g["point"]
                if feats is not None and sample.t >= p["shown"] and (f := gz.predict([sample])) is not None:
                    raw = f[0] + gz.offset
                    p["frames"].append({"t": sample.t, "guess": [float(f[0][0]), float(f[0][1])], "raw": [float(raw[0]), float(raw[1])], "feats": feats, "frame": sample})
                if now - p["shown"] >= LOOK:
                    next_point(now)
                    if g["point"] is None:
                        end_measure()
                        if "play" in rounds:
                            phase, g["break_until"] = "break", now + BREAK
                        else:
                            done()
                if phase == "measure":
                    p = g["point"]
                    held = now - p["shown"]
                    radius = 6 + 24 * max(0.0, 1 - held / WINDOW[0])
                    pygame.draw.circle(screen, TARGET, p["target"], radius)
                    pygame.draw.circle(screen, BACKGROUND, p["target"], 2)
                    text([f"point {len(g['measure']) + 1}/{len(POINTS)}   look at it until it is gone   {status}"], h - 40, colour=DIM)

            elif phase == "break":
                text(["now the circles: bring the pointer in with your eyes"], h // 2 - 20, big)
                if now >= g["break_until"]:
                    phase = "play"
                    begin_play(now)

            elif phase == "play":
                pointer, gaze_f, head_f = g["pointer"]
                while g["queue"] and now >= g["next_at"] and len(g["up"]) < g["want"]:
                    spawn(now)
                guess = gz.predict([sample]) if feats is not None else None
                if guess is not None:
                    t = sample.t
                    pointer.update(t, gaze_f(guess[0], t), head_f((feats["head_x"], feats["head_y"]), t))
                    gp = [float(guess[0][0]), float(guess[0][1])]
                    up = g["up"]
                    if up:
                        up[nearest(gp, [(c["x"], c["y"], c["r"]) for c in up])]["frames"].append({"t": t, "guess": gp, "feats": feats})
                    at = pointer_at()
                    for c in list(up):
                        if at is not None and math.hypot(at[0] - c["x"], at[1] - c["y"]) <= c["r"]:
                            c["inside"] = t if c["inside"] is None else c["inside"]
                            if t - c["inside"] >= DWELL:
                                finish(c, t, True)
                        else:
                            c["inside"] = None
                for c in list(g["up"]):
                    if now - c["shown"] >= TIMEOUT:
                        finish(c, now, False)
                if not g["up"] and not g["queue"]:
                    done()
                else:
                    for c in g["up"]:
                        pygame.draw.circle(screen, TARGET, (c["x"], c["y"]), c["r"], 3)
                        if c["inside"] is not None:
                            k = min(1.0, (now - c["inside"]) / DWELL)
                            pygame.draw.circle(screen, TARGET, (c["x"], c["y"]), max(2, int(c["r"] * k)))
                        pygame.draw.circle(screen, TARGET, (c["x"], c["y"]), 3)
                    flash = g["flash"]
                    if flash is not None and now < flash[1]:
                        surf = font.render("taken" if flash[0] else "missed", True, HIT if flash[0] else WARN)
                        screen.blit(surf, (flash[2] - surf.get_width() / 2, flash[3] - surf.get_height() / 2))
                    if pointer.shown is not None and not pointer.released:
                        pygame.draw.circle(screen, GAZE, (float(pointer.shown[0]), float(pointer.shown[1])), 10, 3)
                    played = [x for x in g["lines"] if x["round"] == "play"]
                    taken = sum(1 for x in played if x["hit"])
                    text([f"circle {len(played)}/{total}   taken {taken}   {status}   Esc: quit"], h - 40, colour=DIM)

            else:  # done
                text(summary, 40)
                text(["Space: play again   Esc: quit"], h - 60, colour=DIM)

            pygame.display.flip()
            clock.tick(60)
    finally:
        if phase in ("measure", "break", "play"):
            if phase == "measure" and not g["measured"]:
                if g["point"] is not None and len(g["point"]["frames"]) >= MIN_SAMPLES:
                    g["measure"].append(g["point"])
                end_measure()
            save_game()
        camera.stop()
        camera.join(timeout=2)
        pygame.quit()

