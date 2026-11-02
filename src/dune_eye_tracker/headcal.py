"""`dune-eye-tracker calibrate-head`: what turning the head does to the eyes' guess.

After `calibrate` (the head still), on the same monitor: 5 points, and on each the eyes
stay on it while the head turns slowly (left, right, up, down, as it says). The eyes
turn the other way to keep looking at the point, which the eyes model takes for the gaze
moving; `model.fit_head` learns that back from the head. It reports, on each point left
out, how far off the guesses are with the eyes alone and with the head, and `run` uses
it only when it is better. Saved in calibration.json (`head_fix`, `head_samples`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from . import model
from .features import BLINK, Camera
from .measure import BACKGROUND, DIM, TARGET, TEXT, WARN, calibration_path
from .track import eyes_model

POINTS = [(0.5, 0.5), (0.2, 0.25), (0.8, 0.25), (0.2, 0.75), (0.8, 0.75)]
SETTLE = 1.0  # s on a point, head still, before it counts
TURNS = ("left", "right", "up", "down")
TURN_TIME = 1.6  # s for each: about 6.5 s a point


def fit(data: dict, samples: list[dict]) -> tuple[model.HeadFix, float, float]:
    """The head fix from `samples` (as `calibrate` saves them) over `data`'s eyes model."""
    w, h = data["size"]
    eyes = eyes_model(data)
    rows = [s for s in samples if s["feats"]["open"] >= BLINK]
    targets = np.array([[s["target"][0] * w, s["target"][1] * h] for s in rows])
    groups = np.array([s["point"] for s in rows])
    return model.fit_head(eyes, [s["feats"] for s in rows], targets, groups, data["head"])


def run(args) -> int:
    path = calibration_path()
    if not path.exists():
        print(f"no calibration at {path}: run `dune-eye-tracker calibrate` first", file=sys.stderr)
        return 1
    data = json.loads(path.read_text())
    if not data.get("head"):
        print("this calibration has no head place: run `dune-eye-tracker calibrate` again", file=sys.stderr)
        return 1

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

    def text(lines, y, f=font, colour=TEXT):
        for line in lines:
            surf = f.render(line, True, colour)
            screen.blit(surf, ((screen.get_width() - surf.get_width()) // 2, y))
            y += surf.get_height() + 8

    asked_fullscreen = not args.fullscreen or shutil.which("dune") is None
    phase, index, held, last_tick, last_seq = "wait", 0, 0.0, time.monotonic(), 0
    samples: list[dict] = []
    note, report = "", []
    try:
        while True:
            now = time.monotonic()
            dt, last_tick = now - last_tick, now
            sample = camera.sample()
            fresh = sample.feats is not None and sample.seq != last_seq
            if fresh:
                last_seq = sample.seq
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return 0
                if event.type == pygame.WINDOWFOCUSGAINED and not asked_fullscreen:
                    asked_fullscreen = True
                    subprocess.run(["dune", "action", "fullscreen"], check=False)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and phase == "wait":
                    if list(screen.get_size()) != list(data["size"]):
                        note = f"this window is {screen.get_width()}x{screen.get_height()}: it must be fullscreen on {data.get('output')} ({data['size'][0]}x{data['size'][1]})"
                    elif sample.feats is None:
                        note = "no face yet: look at the screen, light on your face"
                    else:
                        phase, index, held, samples, note = "points", 0, 0.0, [], ""

            w, h = screen.get_size()
            screen.fill(BACKGROUND)
            if camera.error:
                text([camera.error, "Esc: quit"], h // 2 - 30)
            elif phase == "wait":
                text(["dune-eye-tracker: the head", ""], h // 3, big)
                text(
                    [
                        "Sit as for `calibrate`. On each orange point keep the eyes on it",
                        "and turn the head slowly as it says: left, right, up, down",
                        "(a little, as you would to look around the screen).",
                        f"{len(POINTS)} points, about {len(POINTS) * (SETTLE + TURN_TIME * len(TURNS)):.0f} s.",
                        "",
                        "Space: start     Esc: quit",
                    ],
                    h // 3 + 70,
                )
                if note:
                    text([note], h // 3 + 300, colour=WARN)
            elif phase == "points":
                held += dt
                total = SETTLE + TURN_TIME * len(TURNS)
                if held >= total:
                    index, held = index + 1, 0.0
                    if index == len(POINTS):
                        fix, alone, fixed = fit(data, samples)
                        used = fixed < alone
                        data |= {"head_samples": samples, "head_fix": fix.to_json() | {"alone": alone, "fixed": fixed, "used": used}}
                        path.write_text(json.dumps(data))
                        report = [
                            f"{len(samples)} samples, the head turning, each point left out in turn:",
                            f"eyes alone       {alone:5.0f} px off",
                            f"with the head    {fixed:5.0f} px off",
                            "",
                            "used by `run`" if used else "NOT used: no better than the eyes alone",
                            f"saved: {path}",
                        ]
                        print("\n".join(report), flush=True)
                        phase = "done"
                        continue
                tx, ty = POINTS[index]
                pygame.draw.circle(screen, TARGET, (tx * w, ty * h), 8)
                pygame.draw.circle(screen, BACKGROUND, (tx * w, ty * h), 2)
                if held < SETTLE:
                    say = "eyes on the point, head still"
                else:
                    say = f"keep looking at it, turn the head slowly: {TURNS[min(int((held - SETTLE) / TURN_TIME), len(TURNS) - 1)]}"
                    if fresh and sample.feats["open"] >= BLINK:
                        samples.append({"point": index, "target": [tx, ty], "feats": sample.feats})
                text([say], int(ty * h) + (40 if ty < 0.5 else -70))
                face = "" if sample.feats else "   no face"
                text([f"point {index + 1}/{len(POINTS)}{face}"], h - 40, colour=DIM if sample.feats else WARN)
            else:
                text(report, h // 3)
                text(["Esc: quit"], h - 60, colour=DIM)
            pygame.display.flip()
            clock.tick(60)
    finally:
        camera.stop()
        camera.join(timeout=2)
        pygame.quit()
        if camera.is_alive():
            os._exit(0)
