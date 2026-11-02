"""`dune-eye-tracker measure`: how well the camera knows where you look.

A window fullscreen on the monitor you use (dune is asked once it has focus) shows 15
points to calibrate on (5 across the screen, 3 down), then 10 others to test on: look
at each one until it is gone. The head stays where it was when Space was pressed: when
it turns or moves away from there, the point waits (and says so) until it is back, so
only the eyes move between points. Then, for each feature set (`eyes`, `head`, `both`,
see `model.SETS`), how far off it is on the test points, and a live dot where the best
one thinks you look (Esc quits). The run (only numbers, no images) is saved so
`analyze` can look at it again.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import model
from .features import BLINK, Camera

CALIBRATION = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.06, 0.28, 0.5, 0.72, 0.94)]
TEST = [(x, y) for y in (0.3, 0.7) for x in (0.17, 0.39, 0.61, 0.83)] + [(0.5, 0.2), (0.5, 0.8)]

SHOW = 2.2  # s a point is shown (with the head still)
SETTLE = 0.9  # s before its samples count (the eyes get there, then steady)

# How far the head may go from where it was at Space: turned (the x and y of the face's
# forward axis, 0.05 is about 3°) or moved (cm).
HEAD_TURN = 0.05
HEAD_SHIFT = 1.5

BACKGROUND = (24, 24, 27)
TEXT = (228, 228, 231)
DIM = (113, 113, 122)
TARGET = (249, 115, 22)
GAZE = (251, 191, 36)
WARN = (248, 113, 113)


def state_dir() -> Path:
    state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state / "dune-eye-tracker"


def runs_dir() -> Path:
    return state_dir() / "runs"


def calibration_path() -> Path:
    """Where `calibrate` keeps its samples, for `run`."""
    return state_dir() / "calibration.json"


def pointer_output() -> str | None:
    """The monitor the pointer is on (`dune pointer where`), None without dune."""
    try:
        out = subprocess.run(["dune", "pointer", "where"], capture_output=True, text=True, timeout=2).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out[2] if len(out) == 3 else None


class HeadGuard:
    """Where the head was when measuring started, and whether it has gone from there."""

    def __init__(self, recent: list[dict[str, float]]):
        self.ref = {k: float(np.median([f[k] for f in recent])) for k in ("head_x", "head_y", "pos_x", "pos_y")}

    def off(self, feats: dict[str, float]) -> str | None:
        """What is off ("turned", "moved"), or None while the head is where it was."""
        if max(abs(feats[k] - self.ref[k]) for k in ("head_x", "head_y")) > HEAD_TURN:
            return "turned"
        if max(abs(feats[k] - self.ref[k]) for k in ("pos_x", "pos_y")) > HEAD_SHIFT:
            return "moved"
        return None


def analyze(run: dict, distance_cm: float | None = None, width_cm: float | None = None) -> tuple[dict, str]:
    """The models fitted on the calibration samples of `run`, and a table of how they do
    on its test samples."""
    w, h = run["size"]

    def part(name):
        rows = [s for s in run[name] if s["feats"]["open"] >= BLINK]
        targets = np.array([[s["target"][0] * w, s["target"][1] * h] for s in rows])
        return [s["feats"] for s in rows], targets, np.array([s["point"] for s in rows])

    cal, test = part("calibration"), part("test")
    lines = [f"{len(cal[0])} calibration samples, {len(test[0])} test samples, screen {w}x{h}"]
    if run.get("waited"):
        lines[0] += f", {run['waited']:.0f} s waiting for the head"
    angle = distance_cm and width_cm
    width = 14 if angle else 7

    def cell(px):
        if angle:
            return f"{px:5.0f} ({model.degrees(px, w / width_cm, distance_cm):4.1f})"
        return f"{px:5.0f}"

    heads = ["median", "x", "y", "worst", "jitter", f"smooth {model.SMOOTH}"]
    lines.append(f"{'set':6} " + " ".join(f"{t:>{width}}" for t in heads) + "  ridge" + ("   (px, deg)" if angle else "   (px)"))
    models = {}
    for name in model.SETS:
        m = model.fit(*cal, name)
        r = model.evaluate(m, *test, bounds=(w, h))
        models[name] = (m, r)
        cells = [r.median, r.x, r.y, r.worst, r.jitter, r.smooth]
        lines.append(f"{name:6} " + " ".join(f"{cell(c):>{width}}" for c in cells) + f"  {m.lam:g}")
    return models, "\n".join(lines)


def run(args) -> int:
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
    pygame.display.set_caption("dune-eye-tracker")
    font = pygame.font.SysFont("JetBrains Mono,DejaVu Sans Mono,monospace", 22)
    big = pygame.font.SysFont("JetBrains Mono,DejaVu Sans Mono,monospace", 30)
    clock = pygame.time.Clock()

    calibrate = args.command == "calibrate"  # only the calibration, saved for `run`
    output: str | None = None
    asked_fullscreen = not args.fullscreen or shutil.which("dune") is None
    phase = "wait"  # wait, calibration, test, done
    points: list = []
    index, held, last_tick = 0, 0.0, time.monotonic()
    last_seq = 0
    recent: deque = deque(maxlen=30)  # the faces of the last second, for the head's place
    guard: HeadGuard | None = None
    samples: dict = {"calibration": [], "test": []}
    waited = 0.0
    report, best, smooth = "", None, None
    note = ""

    def text(lines, y, f=font, colour=TEXT):
        for line in lines:
            surf = f.render(line, True, colour)
            screen.blit(surf, ((screen.get_width() - surf.get_width()) // 2, y))
            y += surf.get_height() + 8

    try:
        while True:
            now = time.monotonic()
            dt, last_tick = now - last_tick, now
            sample = camera.sample()
            fresh = sample.feats is not None and sample.seq != last_seq
            if fresh:
                last_seq = sample.seq
                recent.append(sample.feats)

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return 0
                if event.type == pygame.WINDOWFOCUSGAINED and not asked_fullscreen:
                    # dune ignores a fullscreen request before the window is a tile: ask it now.
                    asked_fullscreen = True
                    subprocess.run(["dune", "action", "fullscreen"], check=False)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if phase == "wait":
                        if len(recent) < 10:
                            note = "no face yet: look at the screen, light on your face"
                        elif calibrate and not pygame.mouse.get_focused():
                            note = "move the pointer onto this screen first: it is the one the tracker drives"
                        elif calibrate and not (output := pointer_output()):
                            note = "cannot ask dune where the pointer is (dune pointer where)"
                        else:
                            guard = HeadGuard(list(recent))
                            phase, points, index, held, waited, note = "calibration", CALIBRATION, 0, 0.0, 0.0, ""
                    elif phase == "done":
                        phase, samples, smooth, best = "wait", {"calibration": [], "test": []}, None, None

            w, h = screen.get_size()
            screen.fill(BACKGROUND)
            if camera.error:
                text([camera.error, "Esc: quit"], h // 2 - 30)
                pygame.display.flip()
                clock.tick(30)
                continue

            face = "face: yes" if sample.feats else "face: NO (look at the screen, light on your face)"
            status = f"{camera.fps:4.1f} fps   {face}"

            if phase == "wait":
                title = "calibrating the pointer" if calibrate else "measuring"
                text([f"dune-eye-tracker: {title}", ""], h // 3, big)
                text(
                    [
                        "Sit as you usually do, glasses on, and look at the middle of the screen.",
                        "Space fixes where your head is: from then on only the eyes move.",
                        "Follow each orange point with your eyes until it is gone; if the head",
                        "turns or moves, the point waits until it is back.",
                        f"{len(CALIBRATION)} points to calibrate (about 40 s)." if calibrate else f"{len(CALIBRATION)} points to calibrate, then {len(TEST)} to test (about a minute).",
                        "Keep the pointer on this screen: it is the one the tracker will drive." if calibrate else "",
                        "",
                        "Space: start     Esc: quit",
                    ],
                    h // 3 + 70,
                )
                if note:
                    text([note], h // 3 + 340, colour=WARN)
                text([status], h - 60, colour=DIM)

            elif phase in ("calibration", "test"):
                off = guard.off(sample.feats) if sample.feats else "lost"
                if off is None:
                    held += dt
                else:
                    waited += dt
                if held >= SHOW:
                    index, held = index + 1, 0.0
                    if index == len(points):
                        if phase == "calibration" and calibrate:
                            data = {"size": [w, h], "output": output, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "waited": waited, "head": guard.ref, "calibration": samples["calibration"]}
                            path = calibration_path()
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps(data))
                            print(f"calibrated on {output} ({w}x{h}), {len(samples['calibration'])} samples: {path}")
                            return 0
                        if phase == "calibration":
                            phase, points, index = "test", TEST, 0
                        else:
                            phase = "done"
                            data = {"size": [w, h], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "waited": waited, "head": guard.ref, **samples}
                            path = runs_dir() / f"{data['time']}.json"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps(data))
                            models, report = analyze(data, args.distance_cm, args.width_cm)
                            name = min(models, key=lambda n: models[n][1].median)
                            report += f"\n\nbest: {name}   saved: {path}"
                            print(report)
                            best = models[name][0]
                        continue
                tx, ty = points[index]
                radius = 6 + 24 * max(0.0, 1 - held / SETTLE)
                colour = TARGET if off is None else DIM
                pygame.draw.circle(screen, colour, (tx * w, ty * h), radius)
                pygame.draw.circle(screen, BACKGROUND, (tx * w, ty * h), 2)
                if off is None and held >= SETTLE and fresh:
                    samples[phase].append({"point": index, "target": [tx, ty], "feats": sample.feats})
                if off == "turned":
                    text(["head turned: back to where it was, only the eyes move"], h // 2 - 15, colour=WARN)
                elif off == "moved":
                    text(["head moved: back to where it was"], h // 2 - 15, colour=WARN)
                elif off == "lost":
                    text(["no face: look at the screen"], h // 2 - 15, colour=WARN)
                label = "calibrating" if phase == "calibration" else "testing"
                text([f"{label} {index + 1}/{len(points)}   {status}"], h - 40, colour=DIM)

            else:  # done: the report, and where the best model thinks you look
                text(report.split("\n"), 60)
                text(["The yellow dot is where it thinks you look.   Space: measure again   Esc: quit"], h - 60, colour=DIM)
                if best and fresh and sample.feats["open"] >= BLINK:
                    p = best.predict([sample.feats], (w, h))[0]
                    smooth = p if smooth is None else smooth * 0.75 + p * 0.25
                if smooth is not None:
                    pygame.draw.circle(screen, GAZE, (float(smooth[0]), float(smooth[1])), 14, 3)

            pygame.display.flip()
            clock.tick(60)
    finally:
        camera.stop()
        pygame.quit()


def analyze_file(args) -> int:
    data = json.loads(Path(args.file).read_text())
    _, report = analyze(data, args.distance_cm, args.width_cm)
    print(report)
    return 0
