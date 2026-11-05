"""`dune-eye-tracker record`: known points looked at, every frame's landmarks and pose kept;
and `calibrate`, its first block alone, saved as the calibration `run` uses (`gaze.npz`).

A window fullscreen (dune is asked once it has focus). Space starts three blocks, about
two and a half minutes, sitting as you usually do, the head free:

- calibration: 28 points (7 x 4, the edges in), `LOOK` s each;
- head: 5 points, `HEAD_LOOK` s each, the eyes kept on the point while the head turns
  slowly a little (as to read across the screen);
- grid: the 24 points of dune-eye-tracker's game (a 6 x 4 grid, the edges in).

No pointer is shown. The camera at 1920x1080; the iris is also found in the full frame
(`iris`). Saved to `~/.local/state/dune-eye-tracker/recordings/<time>.npz`: numbers only, unless
`--keep-eyes` (the two eye crops of every frame, grey, 160x100: so the iris finder can be
tuned again on them; they are images of your eyes). `dune-eye-tracker eval` fits the models on it.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from .camera import Camera
from .recording import BLOCKS, build, recordings_dir

# 7 x 4 with the edges: the gaze is squeezed there and a map with a bend needs points to
# learn it (on 3 recordings, 15 points: 189 px; ~27 with more at the edges: 158, quadratic)
CALIBRATION = [(x, y) for y in (0.03, 0.36, 0.64, 0.97) for x in (0.02, 0.12, 0.29, 0.5, 0.71, 0.88, 0.98)]
HEAD = [(0.5, 0.5), (0.2, 0.25), (0.8, 0.25), (0.2, 0.75), (0.8, 0.75)]
GRID = [(x, y) for y in (0.06, 0.3, 0.7, 0.94) for x in (0.04, 0.17, 0.39, 0.61, 0.83, 0.96)]
LOOK = 1.8  # s a calibration or grid point shows
HEAD_LOOK = 4.5  # s a head-moving point shows
BREAK = 2.5  # s between blocks
APART = 350.0  # px at least between a point and the next, when one is left that is

BACKGROUND = (24, 24, 27)
TEXT = (228, 228, 231)
DIM = (113, 113, 122)
TARGET = (249, 115, 22)
WARN = (248, 113, 113)

SAYS = {
    "calibration": "calibration: look at each point until it is gone",
    "head": "now keep your eyes on the point and turn the head slowly, a little, around it",
    "grid": "now the grid: look at each point until it is gone",
}


def order(points, size, rng: np.random.Generator, apart: float = APART) -> list[int]:
    """The points in a random order, each `apart` px from the last when one is left that is."""
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


def pointer_output() -> str | None:
    """The monitor the pointer is on (`dune pointer where`), None without dune."""
    try:
        out = subprocess.run(["dune", "pointer", "where"], capture_output=True, text=True, timeout=2).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out[2] if len(out) == 3 else None


def plan(size, rng: np.random.Generator, blocks=("calibration", "head", "grid")) -> list[tuple[str, int, tuple[float, float], float]]:
    """The points to show: (block, number in its block, target px, s shown)."""
    w, h = size
    out = []
    every = {"calibration": (CALIBRATION, LOOK), "head": (HEAD, HEAD_LOOK), "grid": (GRID, LOOK)}
    for block in blocks:
        points, look = every[block]
        for i in order(points, size, rng):
            out.append((block, i, (points[i][0] * w, points[i][1] * h), look))
    return out


def calibrated(kept: list[dict], meta: dict, output: str) -> str:
    """`calibrate`'s points as the calibration (`gaze.npz`), fitted first (a bad one says so
    here, not in `run`); the one it replaces kept beside it with its time (its clicks)."""
    from . import gaze

    rec = build(kept, meta | {"output": output})
    g = gaze.Gaze(rec)
    path = gaze.store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = gaze.Recording.load(path)
        path.replace(path.with_name(f"gaze.{old.meta.get('time') or 'old'}.npz".replace(":", "-")))
    rec.save(path)
    return f"calibrated on {output} ({meta['screen'][0]}x{meta['screen'][1]}, {len(np.unique(rec.point))} points, {g.model.name}): {path}"


def run(args) -> int:
    calibrate = args.command == "calibrate"  # the calibration block alone, for `run`
    blocks = ("calibration",) if calibrate else ("calibration", "head", "grid")
    keep_eyes = getattr(args, "keep_eyes", False)
    size_arg = getattr(args, "size", None) or (1920, 1080)
    output: str | None = None
    os.environ.setdefault("SDL_VIDEODRIVER", "wayland,x11")
    os.environ.setdefault("SDL_VIDEO_WAYLAND_WMCLASS", "dune-eye-tracker")
    import pygame

    try:
        camera = Camera(args.device, size_arg, keep_crops=keep_eyes)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    camera.start()
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption(f"dune-eye-tracker {'calibrate' if calibrate else 'record'}")
    font = pygame.font.SysFont("JetBrains Mono,DejaVu Sans Mono,monospace", 22)
    big = pygame.font.SysFont("JetBrains Mono,DejaVu Sans Mono,monospace", 30)
    clock = pygame.time.Clock()
    asked = not args.fullscreen or shutil.which("dune") is None

    phase = "wait"  # wait, show, break, done
    steps: list = []
    step, shown, until, size = 0, 0.0, 0.0, (0, 0)
    kept: list[dict] = []
    fps: list[float] = []
    last_seq, note, saved = 0, "", ""

    def text(rows, y, f=font, colour=TEXT):
        for row in rows:
            surf = f.render(row, True, colour)
            screen.blit(surf, ((screen.get_width() - surf.get_width()) // 2, y))
            y += surf.get_height() + 8

    def save() -> str:
        if not kept:
            return ""
        meta = {"screen": list(size), "image": list(kept[-1]["image"]), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "look": {"calibration": LOOK, "head": HEAD_LOOK, "grid": LOOK}, "complete": step >= len(steps),
                "fps": float(np.median(fps)) if fps else None, "eyes_kept": bool(keep_eyes)}
        if calibrate:
            return calibrated(kept, meta, output)
        path = recordings_dir() / f"{meta['time']}.npz"
        build(kept, meta).save(path)
        return str(path)

    try:
        while True:
            now = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return 0
                if event.type == pygame.WINDOWFOCUSGAINED and not asked:
                    asked = True
                    subprocess.run(["dune", "action", "fullscreen"], check=False)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and phase in ("wait", "done"):
                    if camera.frame().landmarks is None:
                        note = "no face yet: look at the screen, light on your face"
                    elif calibrate and not pygame.mouse.get_focused():
                        note = "move the pointer onto this screen first: it is the one the tracker drives"
                    elif calibrate and not (output := pointer_output()):
                        note = "cannot ask dune where the pointer is (dune pointer where)"
                    else:
                        size = screen.get_size()
                        steps, step, kept, note = plan(size, np.random.default_rng(), blocks), 0, [], ""
                        phase, until = "break", now + BREAK
                        pygame.mouse.set_visible(False)

            w, h = screen.get_size()
            screen.fill(BACKGROUND)
            if camera.error:
                text([camera.error, "Esc: quit"], h // 2 - 30)
                pygame.display.flip()
                clock.tick(30)
                continue
            fr = camera.frame()
            status = f"{camera.fps:4.1f} fps   face: {'yes' if fr.landmarks is not None else 'NO'}   window {w}x{h}"

            if phase in ("show", "break") and fr.seq != last_seq:
                last_seq = fr.seq
                if phase == "show" and fr.landmarks is not None and fr.t >= shown:
                    block, i, target, _ = steps[step]
                    row = {"landmarks": fr.landmarks, "matrix": fr.matrix, "t": fr.t, "block": BLOCKS.index(block),
                           "point": i, "target": target, "since": fr.t - shown, "image": fr.image}
                    if fr.eyes is not None:
                        row |= {"limbus": fr.eyes["limbus"], "dark": fr.eyes["dark"], "maps": fr.eyes["maps"]}
                        if keep_eyes:
                            row["crops"] = fr.eyes["crops"]
                    kept.append(row)
                    fps.append(camera.fps)

            if phase == "wait":
                text([f"dune-eye-tracker: {'calibrating the pointer' if calibrate else 'record'}", ""], h // 3 - 40, big)
                says = [
                    f"{len(CALIBRATION)} points to calibrate: look at each orange point until it is gone. About a minute.",
                    "Keep the pointer on this screen: it is the one the tracker will drive.",
                    "From then on your clicks keep it calibrated (look at the pointer when you click).",
                ] if calibrate else [
                    f"{len(CALIBRATION)} points to calibrate, {len(HEAD)} to keep looking at while the head turns a little,",
                    f"then {len(GRID)} to test. Look at each orange point until it is gone. About 2.5 minutes.",
                ]
                text(
                    ["Sit as you usually do, glasses on; the head may move.", *says,
                     "Only numbers are kept (the face's landmarks), no images.", "", "Space: start     Esc: quit"],
                    h // 3 + 30,
                )
                if note:
                    text([note], h // 3 + 260, colour=WARN)
                text([status], h - 60, colour=DIM)

            elif phase == "break":
                text([SAYS[steps[step][0]]], h // 2 - 20, big)
                if now >= until:
                    phase, shown = "show", now

            elif phase == "show":
                block, i, target, look = steps[step]
                held = now - shown
                pygame.draw.circle(screen, TARGET, target, 6 + 24 * max(0.0, 1 - held / 0.6))
                pygame.draw.circle(screen, BACKGROUND, target, 2)
                if block == "head":
                    text(["eyes on the point; turn the head slowly, a little"], int(target[1]) + 40 if target[1] < h / 2 else int(target[1]) - 70, colour=DIM)
                text([f"{block} {sum(1 for s in steps[:step] if s[0] == block) + 1}/{sum(1 for s in steps if s[0] == block)}   {status}"], h - 40, colour=DIM)
                if held >= look:
                    step += 1
                    if step == len(steps):
                        try:
                            saved = save()
                        except Exception as e:  # too few points with a face: said, nothing replaced
                            saved = f"not saved: {e}"
                        print(saved if calibrate else f"{len(kept)} frames: {saved}", flush=True)
                        if calibrate:
                            return 0
                        phase = "done"
                        pygame.mouse.set_visible(True)
                    elif steps[step][0] != block:
                        phase, until = "break", now + BREAK
                    else:
                        shown = now

            else:
                text([f"saved {len(kept)} frames:", saved, "", "dune-eye-tracker eval   (Space: record again, Esc: quit)"], h // 3)

            pygame.display.flip()
            clock.tick(60)
    finally:
        if phase in ("show", "break") and kept and not calibrate:  # a calibration is all or nothing
            print(f"stopped early, {len(kept)} frames: {save()}", flush=True)
        camera.stop()
        camera.join(timeout=2)
        pygame.quit()
