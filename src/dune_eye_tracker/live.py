"""`dune-eye-tracker live`: where the model thinks you look, as you look.

A window fullscreen. The model (`model`: geometric on the iris's dark centroid found in the
1080p frame, a quadratic map with the 28-point calibration) is fitted on the calibration points of the last 1080p recording (`--from` one
in particular), or on a quick calibration made here (C: the 15 points, kept in memory only).
A yellow dot is the gaze filtered (One Euro, steadier than dune-eye-tracker's), a faint one
the frame's own guess; both kept on the screen.

T measures it: a point of the recording grid shows (all 24 in turn, shuffled, before any
again); look at it until it is gone; its miss (the median guess 0.6..1.5 s after it showed)
adds to the session's, shown with its 95% interval. R adjusts a saved calibration to now: 9
points looked at, the model's median miss on them taken off every guess (vertically only:
on a saved calibration the miss was a shift down of ~130 px; 9 points took 187 px to ~121
in replay, both axes ~129). X clears the tests shown. Esc quits. The tracker's `run` has to
be off (the camera).

The session is saved as a recording (`<time>-live.npz`: the last C's calibration as its
calibration block, every T as a grid point; numbers only, the eye crops too with
`--keep-eyes`), so `dune-eye-tracker eval` can take it apart like any other.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import evaluate
from .camera import Camera
from .models import Geometric
from .record import BACKGROUND, CALIBRATION, DIM, GRID, LOOK, TARGET, TEXT, WARN, order
from .recording import BLOCKS, Recording, WINDOWS, all_recordings, build, recordings_dir

GAZE = (251, 191, 36)
RAW = (113, 113, 122)
HIT = (74, 222, 128)
SOURCE = "dark"
# One Euro (Casiez 2012), as dune-eye-tracker's gaze: Hz when still, and how it opens with speed
# (beta 0.004: slower to follow speed than the tracker's 0.01, as asked: steadier)
MINCUTOFF, BETA, DCUTOFF = 0.6, 0.004, 1.0
QUADRATIC_FROM = 24  # calibration points from which the map gets its bend (`Geometric` order 2)
ADJUST = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.08, 0.5, 0.92)]  # R: the points looked at
ADJUST_AXES = (False, True)  # which of the shift's axes is taken off (x, y): y only (replay)


def shift(guesses: np.ndarray, targets: np.ndarray, axes=ADJUST_AXES) -> np.ndarray:
    """The model's constant miss on the adjust points (each point's median guess less its
    target; their median), only on `axes`."""
    d = np.median(np.asarray(guesses, float) - np.asarray(targets, float), axis=0)
    return np.where(axes, d, 0.0)



def model(points: int) -> Geometric:
    """Geometric on the dark centroid; with enough calibration points (the 28 of now) a
    quadratic map, which bends at the edges where the gaze is squeezed."""
    order = 2 if points >= QUADRATIC_FROM else 1
    return Geometric(f"geometric{', quadratic' if order == 2 else ''} [{SOURCE}]", order, source=SOURCE)


class OneEuro:
    """A One Euro filter on a 2D point."""

    def __init__(self, mincutoff: float = MINCUTOFF, beta: float = BETA, dcutoff: float = DCUTOFF):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x = self.dx = None
        self.t = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dcutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mincutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x, self.t = a * x + (1 - a) * self.x, t
        return self.x


def one_frame(fr, screen) -> Recording | None:
    """A camera frame as a one-frame recording (what the models take), None without a
    face or the full-frame iris."""
    if fr.landmarks is None or fr.eyes is None:
        return None
    return Recording(
        fr.landmarks[None].astype(float), fr.matrix[None].astype(float), np.array([fr.t]), np.zeros(1, int), np.zeros(1, int),
        np.zeros((1, 2)), np.zeros(1), {"screen": list(screen), "image": list(fr.image)}, "live",
        limbus=fr.eyes["limbus"][None], dark=fr.eyes["dark"][None], maps=fr.eyes["maps"][None],
    )


def fitted(rec: Recording):
    idx = evaluate.frames(rec, "calibration")
    groups = evaluate.groups(rec, idx)
    return model(len(np.unique(groups))).fit(rec, idx, groups)


def last_recording() -> Path | None:
    """The last recording made at 1080p with the full-frame iris and a calibration."""
    for p in reversed(all_recordings()):
        try:
            with np.load(p) as z:
                if "dark" in z and np.count_nonzero(z["block"] == BLOCKS.index("calibration")) > 100:
                    return p
        except (OSError, ValueError):
            continue
    return None


def run(args) -> int:
    path = Path(args.from_file) if args.from_file else last_recording()
    m, source_note = None, f"no model yet: C to calibrate ({len(CALIBRATION)} points)"
    fitted_on = None  # the recording whose calibration the model is fitted on (None: made here)
    if path is not None:
        print(f"fitting on {path.name}'s calibration (the iris found again on its eye crops: a few seconds)", flush=True)
        rec = Recording.load(path)
        m = fitted(rec)
        source_note, fitted_on = f"fitted on {rec.name}'s calibration", rec.name
        fit_screen = rec.screen
    else:
        fit_screen = None

    os.environ.setdefault("SDL_VIDEODRIVER", "wayland,x11")
    os.environ.setdefault("SDL_VIDEO_WAYLAND_WMCLASS", "dune-eye-tracker")
    import pygame

    try:
        camera = Camera(args.device, args.size, keep_crops=args.keep_eyes)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    camera.start()
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("dune-eye-tracker live")
    font = pygame.font.SysFont("JetBrains Mono,DejaVu Sans Mono,monospace", 20)
    big = pygame.font.SysFont("JetBrains Mono,DejaVu Sans Mono,monospace", 30)
    clock = pygame.time.Clock()
    asked = not args.fullscreen or shutil.which("dune") is None
    pygame.mouse.set_visible(False)

    filt = OneEuro()
    gaze = raw = None
    last_seq = 0
    mode = "live"  # live, calibrate, test
    steps: list = []  # calibrate: (number, target); test: one target
    step, shown = 0, 0.0
    kept: list[dict] = []  # the last calibration's frames
    tested: list[dict] = []  # every test's frames
    guesses: list = []  # test: (since, guess)
    misses: list[float] = []  # shown (R clears them)
    all_misses: list[float] = []  # saved
    queue: list[int] = []  # the grid points still to test this round
    offset = np.zeros(2)  # R: taken off every guess
    seen: dict[int, list] = {}  # adjust: each point's guesses (the model's, before the offset)
    adjusts: list[dict] = []  # saved: each R (its points, its shift, the test it came before)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    def row(fr, block: str, point: int, target, since: float) -> dict:
        r = {"landmarks": fr.landmarks, "matrix": fr.matrix, "t": fr.t, "block": BLOCKS.index(block), "point": point,
             "target": target, "since": since, "image": fr.image,
             "limbus": fr.eyes["limbus"], "dark": fr.eyes["dark"], "maps": fr.eyes["maps"]}
        if args.keep_eyes:
            r["crops"] = fr.eyes["crops"]
        return r

    def save() -> Path | None:
        frames = kept + tested
        if not tested and not kept:
            return None
        meta = {"screen": list(screen.get_size()), "image": list(frames[-1]["image"]), "time": started, "live": True,
                "model": source_note, "fitted_on": fitted_on, "adjusts": adjusts, "live_misses": all_misses, "eyes_kept": bool(args.keep_eyes),
                "look": {"calibration": LOOK, "grid": LOOK}, "complete": True}
        path = recordings_dir() / f"{started}-live.npz"
        build(frames, meta).save(path)
        return path
    note = ""
    rng = np.random.default_rng()

    def text(rows, y, f=font, colour=TEXT):
        for row in rows:
            surf = f.render(row, True, colour)
            screen.blit(surf, ((screen.get_width() - surf.get_width()) // 2, y))
            y += surf.get_height() + 6

    try:
        while True:
            now = time.monotonic()
            w, h = screen.get_size()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return 0
                if event.type == pygame.WINDOWFOCUSGAINED and not asked:
                    asked = True
                    subprocess.run(["dune", "action", "fullscreen"], check=False)
                if event.type == pygame.KEYDOWN and mode == "live":
                    if event.key == pygame.K_c:
                        steps = [(i, (CALIBRATION[i][0] * w, CALIBRATION[i][1] * h)) for i in order(CALIBRATION, (w, h), rng)]
                        mode, step, shown, kept, note = "calibrate", 0, now + 1.5, [], ""
                    elif event.key == pygame.K_t and m is not None:
                        if not queue:
                            queue = order(GRID, (w, h), rng)
                        i = queue.pop(0)
                        steps = [(i, (GRID[i][0] * w, GRID[i][1] * h))]
                        mode, step, shown, guesses = "test", 0, now + 0.4, []
                    elif event.key == pygame.K_r and m is not None:
                        steps = [(i, (ADJUST[i][0] * w, ADJUST[i][1] * h)) for i in order(ADJUST, (w, h), rng)]
                        mode, step, shown, seen, note = "adjust", 0, now + 1.5, {}, ""
                    elif event.key == pygame.K_x:
                        misses = []

            screen.fill(BACKGROUND)
            if camera.error:
                text([camera.error, "Esc: quit"], h // 2 - 30)
                pygame.display.flip()
                clock.tick(30)
                continue
            if m is not None and fit_screen is not None and tuple(fit_screen) != (w, h) and mode == "live":
                note = f"the model was fitted on a {fit_screen[0]}x{fit_screen[1]} screen, this window is {w}x{h}: fullscreen, or C"

            fr = camera.frame()
            if fr.seq != last_seq:
                last_seq = fr.seq
                one = one_frame(fr, (w, h))
                if one is not None and m is not None:
                    try:
                        g = m.predict(one, np.array([0]))[0]
                    except Exception:
                        g = None
                    if g is not None and np.all(np.isfinite(g)):
                        if mode == "adjust" and step < len(steps) and fr.t >= shown:
                            lo, hi = WINDOWS["calibration"]
                            if lo <= fr.t - shown <= hi:
                                seen.setdefault(step, []).append(g)
                        g = evaluate.on_screen((g - offset)[None], (w, h))[0]  # never off the monitor
                        raw, gaze = g, filt(g, fr.t)
                        if mode == "test" and fr.t >= shown:
                            guesses.append((fr.t - shown, g))
                if mode == "calibrate" and fr.landmarks is not None and fr.eyes is not None and step < len(steps) and fr.t >= shown:
                    i, target = steps[step]
                    kept.append(row(fr, "calibration", i, target, fr.t - shown))
                if mode == "test" and fr.landmarks is not None and fr.eyes is not None and fr.t >= shown:
                    tested.append(row(fr, "grid", len(all_misses), steps[0][1], fr.t - shown))  # each test its own point

            if mode == "calibrate":
                if step < len(steps):
                    i, target = steps[step]
                    held = now - shown
                    if held >= 0:
                        pygame.draw.circle(screen, TARGET, target, 6 + 24 * max(0.0, 1 - held / 0.6))
                        pygame.draw.circle(screen, BACKGROUND, target, 2)
                    text([f"calibrating {step + 1}/{len(steps)}: look at each point until it is gone"], h - 40, colour=DIM)
                    if held >= LOOK:
                        step, shown = step + 1, now
                else:
                    rec = build(kept, {"screen": [w, h], "image": list(kept[-1]["image"]) if kept else [0, 0]})
                    try:
                        m, fit_screen = fitted(rec), (w, h)
                        source_note, note = f"fitted on a calibration made here ({time.strftime('%H:%M')})", ""
                        fitted_on, offset = None, np.zeros(2)
                        filt, misses = OneEuro(), []
                    except Exception as e:
                        note = f"calibration failed: {e}"
                    mode = "live"

            elif mode == "adjust":
                if step < len(steps):
                    i, target = steps[step]
                    held = now - shown
                    if held >= 0:
                        pygame.draw.circle(screen, TARGET, target, 6 + 24 * max(0.0, 1 - held / 0.6))
                        pygame.draw.circle(screen, BACKGROUND, target, 2)
                    text([f"adjusting {step + 1}/{len(steps)}: look at each point until it is gone"], h - 40, colour=DIM)
                    if held >= LOOK:
                        step, shown = step + 1, now
                else:
                    got = [k for k in range(len(steps)) if len(seen.get(k, [])) >= evaluate.MIN_FRAMES]
                    if len(got) >= (len(steps) + 1) // 2:
                        guess = np.array([np.median(seen[k], axis=0) for k in got])
                        target = np.array([steps[k][1] for k in got])
                        offset = shift(guess, target)
                        full = np.median(guess - target, axis=0)
                        adjusts.append({"before_test": len(all_misses), "targets": target.tolist(), "guesses": guess.tolist(),
                                        "shift": full.tolist(), "applied": offset.tolist()})
                        filt, misses = OneEuro(), []
                        note = f"adjusted on {len(got)} points: {offset[1]:+.0f} px vertically (x was {full[0]:+.0f})"
                    else:
                        note = f"adjust failed: the eyes found on {len(got)} of {len(steps)} points"
                    mode = "live"

            elif mode == "test":
                i, target = steps[0]
                held = now - shown
                if held >= 0:
                    pygame.draw.circle(screen, TARGET, target, 6 + 24 * max(0.0, 1 - held / 0.6))
                    pygame.draw.circle(screen, BACKGROUND, target, 2)
                if held >= LOOK:
                    lo, hi = WINDOWS["grid"]
                    g = [p for s, p in guesses if lo <= s <= hi]
                    d = float(np.linalg.norm(np.median(g, axis=0) - np.array(target))) if len(g) >= evaluate.MIN_FRAMES else None
                    all_misses.append(d)
                    if d is not None:
                        misses.append(d)
                        note = f"that point: {d:.0f} px off"
                    else:
                        note = "that point: no face or eyes"
                    mode = "live"

            if mode == "live":  # not while testing: seeing it, you would look off to put it right
                if raw is not None:
                    pygame.draw.circle(screen, RAW, (float(raw[0]), float(raw[1])), 5)
                if gaze is not None:
                    pygame.draw.circle(screen, GAZE, (float(gaze[0]), float(gaze[1])), 14, 3)

            if mode == "live":
                fps = f"{camera.fps:4.1f} fps   face: {'yes' if fr.landmarks is not None else 'NO'}"
                stats = f"tested {len(misses)}: median {evaluate.med_ci(misses)} px" if misses else "T: look at a point and see how far off it is"
                text([f"{source_note}   {fps}", stats, "C: calibrate here   R: adjust (9 points)   T: test a point   X: clear the tests   Esc: quit"], h - 90, colour=DIM)
                if note:
                    text([note], 30, big, HIT if note.startswith(("that point", "adjusted")) else WARN)

            pygame.display.flip()
            clock.tick(60)
    finally:
        if misses:
            print(f"tested {len(misses)} points: median {evaluate.med_ci(misses)} px", flush=True)
        try:
            if (saved := save()) is not None:
                print(f"session saved: {saved}  (dune-eye-tracker eval {saved})", flush=True)
        except Exception as e:
            print(f"could not save the session: {e}", file=sys.stderr)
        camera.stop()
        camera.join(timeout=2)
        pygame.quit()
