"""`dune-eye-tracker run`: the pointer goes where you look.

No window. The eyes model is fitted on the samples `calibrate` saved; each frame with a
face and the eyes open gives a gaze point, and the head where it points. Both go
through a One Euro filter (steady while still, quick when they move). The gaze only gets
the pointer near (a webcam is ~3° off): when it is far from the pointer (`JUMP` px for
`JUMP_TIME` s) the pointer jumps there; between jumps the head moves it from where it
landed, turned a little for the last few pixels (MAGIC pointing, Zhai 1999). The pointer glides there (`GLIDE`), not a bee. Without a
face, or blinking, the pointer stays. Looking off the screen (`AWAY` px past an edge
for `AWAY_TIME` s: the monitor below, the keyboard) lets go of the pointer, for the
mouse; looking back on it takes it again. Moving the mouse also takes the pointer for the
hand: it stays where the hand leaves it (head still) until the gaze moves `JUMP` px from
where it was then. Left on another monitor by the mouse, it stays there, wherever you
look, until the mouse brings it back to this one. `toggle` starts and stops it.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from OneEuroFilter import OneEuroFilter

from . import model
from .clicks import ClickLearner, Clicks
from .features import BLINK, Camera
from .measure import calibration_path, state_dir
from .snap import Near, Puts

JUMP = 300.0  # px between gaze and pointer that make it jump
JUMP_TIME = 0.15  # s the gaze stays that far first (not one wild frame)
HEAD_GAIN = 1200.0  # px per unit of head turn (0.05, about 3°: 60 px): slow, for the last pixels
AWAY = 150.0  # px past an edge of the screen the gaze is "off it"
AWAY_TIME = 0.3  # s it stays there before the pointer is let go
HAND_TIME = 0.5  # s after the mouse last moved during which the pointer is the hand's
# s: the pointer glides to where it goes (a jump, the head) instead of being put there,
# most of the way (63%) in this time; 0: put there
GLIDE = 0.1
# Turning the head to one side further than looking at the screen takes it (the eyes may
# stay on the screen) steps the strip one tile that way (`dune action focus_left/right`);
# the next step waits for the head to come back `STEP_BACK` towards the middle. How far
# is from the clicks (`step_turns`): past the 1% most turned of them by `STEP_MARGIN`,
# else `STEP_TURN` from the calibration's head. Looking left turns `head_x` up.
STEP_TURN = 0.35
STEP_MARGIN = 0.08
STEP_BACK = 0.15
STEP_CLICKS = 200  # clicks it takes to know how far the head turns looking at the screen
# px per unit of head turn for where the face points on the screen, from the T4 run
# (head_x +0.26 / -0.27 on the left / right corners, 2150 px apart): what the eyes model
# misses when you turn the head to the monitor below, eyes back in the middle.
HEAD_SCREEN = 4000.0

# One Euro filters (Casiez et al. 2012): Hz of the cutoff when still, and how much it
# opens with speed (in the units of each: px for the gaze, the head's axis for the head).
GAZE_FILTER = {"mincutoff": 0.6, "beta": 0.01}
HEAD_FILTER = {"mincutoff": 1.0, "beta": 3.0}
FPS = 27.0


class Pointer:
    """Where the pointer goes, from the gaze (screen px) and the head (`head_x`,
    `head_y`). Looking left turns `head_x` up, looking up `head_y` up: both move the
    pointer the other way on the screen's axes."""

    def __init__(
        self,
        size: tuple[float, float],
        jump: float = JUMP,
        jump_time: float = JUMP_TIME,
        head_gain: float = HEAD_GAIN,
        away: float = AWAY,
        away_time: float = AWAY_TIME,
        head_ref=(0.0, 0.0),
        head_screen: float = HEAD_SCREEN,
        hand_time: float = HAND_TIME,
        glide: float = 0.0,
        step_at: tuple[float, float] | None = None,
    ):
        self.size = np.array(size, dtype=float)
        self.jump, self.jump_time, self.gain = jump, jump_time, head_gain
        self.away, self.away_time = away, away_time
        self.head_ref, self.head_screen = np.asarray(head_ref, dtype=float), head_screen
        self.away_since: float | None = None
        self.released = False  # looking off the screen: the pointer is the mouse's
        self.at: np.ndarray | None = None  # where the pointer is
        self.landed: np.ndarray | None = None  # where the last jump put it
        self.head0: np.ndarray | None = None  # and the head then
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
        self.step = 0  # a tile to step this frame: -1 left, 1 right, 0 none
        self.turned = 0  # stepped that way: the head has to come back first
        # jumps so far: a late answer of where one should have landed (`snap_to`) is only
        # taken for the last one
        self.landings = 0

    def hand(self, t: float, xy):
        """The mouse moved the pointer to `xy` on this monitor (None: on another one)."""
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
                    self.sent = (int(round(self.hand_at[0])), int(round(self.hand_at[1])))
                    self.shown, self.shown_t = self.hand_at.copy(), t
                self.far_since = None
                return None
            self.hand_until, self.hand_gaze = None, gaze.copy()  # stopped: kept until you look away
        if not self.released and self.head_step(head):
            return None  # turned to step: the pointer stays (the eyes model reads the turn too)
        # where you look, the head counted too (the eyes model has it still, where it
        # was at calibration): only for telling whether that is on the screen
        seen = gaze - self.head_screen * (head - self.head_ref)
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
        if self.hand_gaze is not None:
            if np.linalg.norm(gaze - self.hand_gaze) <= self.jump:
                self.far_since = None
                return None  # the gaze is still where it was: the pointer stays as the hand left it
            self.far_since = t if self.far_since is None else self.far_since
            if t - self.far_since < self.jump_time:
                return None
            self.hand_gaze = None
            self.land(gaze, head)
        elif self.at is None:
            self.land(gaze, head)
        elif np.linalg.norm(gaze - self.at) > self.jump:
            self.far_since = t if self.far_since is None else self.far_since
            if t - self.far_since >= self.jump_time:
                self.land(gaze, head)
            else:
                self.follow(head)
        else:
            self.far_since = None
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

    def land(self, gaze, head):
        self.landed = np.clip(gaze, 0, self.size - 1)
        self.landings += 1
        self.at, self.head0, self.far_since = self.landed.copy(), head, None

    def snap_to(self, seq: int, onto) -> bool:
        """Jump `seq` should have landed on `onto` (the button there, `snap.Near`): the
        pointer glides over, the head still moving it from there. Not when another jump
        came since, or the hand has it."""
        if seq != self.landings or self.landed is None or self.at is None:
            return False
        if self.released or self.elsewhere or self.hand_until is not None or self.hand_gaze is not None:
            return False
        moved = np.clip(np.asarray(onto, dtype=float), 0, self.size - 1) - self.landed
        self.landed = self.landed + moved
        self.at = np.clip(self.at + moved, 0, self.size - 1)
        return True

    def follow(self, head):
        self.at = np.clip(self.landed - self.gain * (head - self.head0), 0, self.size - 1)

    def head_step(self, head) -> bool:
        """Whether the head is turned to step the strip (`step` set on the frame it gets
        there). Back from a step, the pointer lands where you look then."""
        left, right = self.step_at
        x = float(head[0])
        if self.turned:
            back = x < left - STEP_BACK if self.turned < 0 else x > right + STEP_BACK
            if not back:
                return True
            self.turned = 0
            self.at = None  # lands on the gaze, with the head as it is now
            return False
        if x >= left or x <= right:
            self.step = self.turned = -1 if x >= left else 1
            self.far_since = None
            return True
        return False


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


def load_calibration(path: Path | None = None) -> tuple[dict, model.Model]:
    """The saved calibration and the eyes model fitted on it."""
    data = json.loads((path or calibration_path()).read_text())
    return data, eyes_model(data)


def head_fix(data: dict) -> model.HeadFix | None:
    """The fix learnt from clicks, else the head calibration's when it did better than the
    eyes alone, else None."""
    if data.get("click_fix"):
        return model.HeadFix.from_json(data["click_fix"])
    fix = data.get("head_fix")
    return model.HeadFix.from_json(fix) if fix and fix.get("used") else None


def eyes_model(data: dict) -> model.Model:
    """The eyes model fitted on a calibration's samples."""
    w, h = data["size"]
    rows = [s for s in data["calibration"] if s["feats"]["open"] >= BLINK]
    targets = np.array([[s["target"][0] * w, s["target"][1] * h] for s in rows])
    groups = np.array([s["point"] for s in rows])
    return model.fit([s["feats"] for s in rows], targets, groups, "eyes")


def calibration_from_run(run_path: str, output: str | None) -> int:
    """`calibrate --from RUN.json`: the calibration points of a `measure` run, for `run`."""
    data = json.loads(Path(run_path).read_text())
    if not output:
        print("say the monitor it was on: --output NAME (as in dune.toml's [output.NAME])", file=sys.stderr)
        return 1
    saved = {k: data[k] for k in ("size", "time", "waited", "head", "calibration") if k in data}
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved | {"output": output}))
    print(f"calibration from {run_path} on {output} ({data['size'][0]}x{data['size'][1]}): {path}")
    return 0


def move(x: int, y: int, output: str | None):
    """`dune pointer move x y output`, straight on dune's socket (27 times a second)."""
    send(f"pointer\tmove\t{x}\t{y}\t{output or ''}")


def action(name: str):
    """`dune action NAME`: what that key of `[keys]` does (`focus_left`...)."""
    send(f"action\t{name}")


def step_turns(data: dict) -> tuple[float, float] | None:
    """`head_x` past which the head steps the strip left / right: further than the 1%
    most turned of the clicks (looking at the screen), by `STEP_MARGIN`. None: too few."""
    xs = [s["feats"]["head_x"] for s in data.get("click_samples") or [] if s.get("feats")]
    if len(xs) < STEP_CLICKS:
        return None
    return float(np.percentile(xs, 99)) + STEP_MARGIN, float(np.percentile(xs, 1)) - STEP_MARGIN


def send(request: str):
    path = os.environ.get("DUNE_SOCKET")
    if not path:
        raise RuntimeError("DUNE_SOCKET is not set: not running inside dune?")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(path)
        s.sendall(request.encode())


def run(args) -> int:
    path = calibration_path()
    if not path.exists():
        print(f"no calibration at {path}: run `dune-eye-tracker calibrate` first", file=sys.stderr)
        return 1
    data, eyes = load_calibration(path)
    size, output = tuple(data["size"]), data.get("output")
    try:
        camera = Camera(args.device)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    camera.start()
    ref = data.get("head") or {}
    fix = head_fix(data)
    near = Near(output, args.snap) if args.snap > 0 and output else None
    if near is not None:
        near.start()
    asked = 0  # the last jump `near` was asked about
    zoom = None  # dune's zoom on this monitor while it is in (a `zoom` line)
    # with the head calibration the gaze already counts the head: "off the screen" too
    pointer = Pointer(
        size,
        head_gain=args.head_gain,
        head_ref=(ref.get("head_x", 0.0), ref.get("head_y", 0.0)),
        head_screen=0.0 if fix else HEAD_SCREEN,
        glide=args.glide,
        step_at=step_turns(data),
    )
    gaze_f, head_f = Filter2(GAZE_FILTER), Filter2(HEAD_FILTER)
    learner = ClickLearner(data, eyes, fix, path)
    clicks = Clicks()
    clicks.start()
    puts = Puts(output) if output else None
    if puts is not None:
        puts.start()
    last_seq, moves, since = 0, 0, time.monotonic()
    if data.get("click_fix"):
        c = data["click_fix"]
        with_head = f", learnt from {c['clicks']} clicks ({c['after']:.0f} px off on the last ones)"
    elif fix:
        with_head = f", the head too ({data['head_fix']['fixed']:.0f} px off, eyes alone {data['head_fix']['alone']:.0f})"
    else:
        with_head = ""
    print(f"driving the pointer on {output} ({size[0]}x{size[1]}); calibrated {data['time']}{with_head}", flush=True)
    if near is not None:
        print(near.error or f"landing on buttons within {args.snap:g} px (dune-a11y, asked at each jump)", flush=True)
    left, right = pointer.step_at
    print(f"a tile left with head_x past {left:+.2f}, right past {right:+.2f}", flush=True)
    try:
        while not stop:
            if camera.error:
                print(camera.error, file=sys.stderr)
                return 1
            sample = camera.sample()
            if sample.seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = sample.seq
            for _, t, x, y in puts.pending() if puts is not None else ():
                # dune-a11y clicked something there (by voice, "click on X"): a click the
                # least trusted (`clicks.WEIGHTS`), you looked at it somewhere; then the
                # pointer onto it, left there as the mouse leaves it, until you look elsewhere
                said = learner.click(t, x, y, output, None, {"by": "name", "button": "left", "hand": pointer.hand_now(t)})
                if args.verbose:
                    print(f"click on {x} {y} (dune-a11y): {said}", flush=True)
                pointer.hand(t, (x, y))
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
                        # magnified: the gaze mapped back, the head and the jumps 1/scale
                        zoom = ctx
                        z = zoom["scale"] if zoom else 1
                        pointer.gain, pointer.jump = args.head_gain / z, JUMP / z
                        if args.verbose:
                            print(f"zoom {'in x' + str(z) if zoom else 'out'}", flush=True)
                    continue
                if change == "move":
                    pointer.hand(t, (x, y) if out == output else None)
                    continue
                said = learner.click(t, x, y, out, ctx, how | {"hand": pointer.hand_now(t)})
                if args.verbose:
                    print(f"click {x} {y} {out}: {said}", flush=True)
            if note := learner.refit_background():  # fits in a thread: the pointer goes on
                print(note, flush=True)
                if learner.fix is not fix:  # the new fix: gaze, and "off the screen", from it
                    fix = learner.fix
                    pointer.head_screen = 0.0
            feats = sample.feats
            if feats is None:
                continue
            learner.seen(sample.t, feats)
            if feats["open"] < BLINK:
                continue
            guess = eyes.predict([feats])  # unclipped: off the screen counts
            if fix:
                guess = fix.correct(guess, [feats])
            gaze = gaze_f(guess[0], sample.t)
            head = head_f((feats["head_x"], feats["head_y"]), sample.t)
            was_released = pointer.released
            if near is not None:
                for seq, onto in near.answers():  # where a jump should have landed
                    if onto is not None and pointer.snap_to(seq, onto) and args.verbose:
                        print(f"jump {seq} onto {onto[0]:.0f} {onto[1]:.0f}", flush=True)
            px = pointer.update(sample.t, unzoom(gaze, zoom), head)
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
    command = "run" if calibration_path().exists() else "calibrate"
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
