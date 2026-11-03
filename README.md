# dune-eye-tracker

The mouse pointer in [dune](https://github.com/dune-wm/dune), moved by where you look, with a webcam.

It all runs on this computer: the camera's images are read and dropped in memory, never
saved nor sent; only numbers are kept (calibration, runs, clicks; with the clicks, the roles
of what could be clicked near them), in `~/.local/state`.
The face model (MediaPipe's `face_landmarker.task`) is downloaded once, by hand.

## Drive the pointer

```sh
git clone https://github.com/dune-wm/dune-eye-tracker && cd dune-eye-tracker
uv tool install --editable .        # dune-eye-tracker in ~/.local/bin
dune-eye-tracker calibrate          # once, and again when you sit differently
```

`calibrate` is the 15 points of `measure` (below) without the test, fullscreen on the
monitor the pointer is on: that is the monitor it drives. It is saved in
`~/.local/state/dune-eye-tracker/calibration.json`.
Then, optionally, `dune-eye-tracker calibrate-head` on the same monitor: 5 points, the
eyes on each while the head turns slowly as it says. It learns what turning the head does
to the eyes' guess (`model.fit_head`: gaze = eyes + b + K·Δhead), reports the error on
each point left out with the eyes alone and with the head, and `run` uses it only when it
is better (then "off the screen" comes from it too). A new `calibrate` drops it.

While `run` goes, **every click teaches it** (look at the pointer when you click): it
hears them from `dune events pointer` (the mouse's, right Ctrl+Shift, dune-voice's "click",
and "click on X" through dune-a11y), keeps the
camera's samples from just before a click on the calibrated monitor when the gaze was
steady, wherever it guessed you looked (it trusts you look at the pointer), and every 5 clicks (the first after 10) fits a fix (head,
gaze and how open the eyes are, `clicks.py`) on them, used when it does no worse on the
last 50 clicks (half of them while there are fewer; up to 15% worse still counts, as the
fix in use has seen most of them). Not all clicks count the same: the mouse's 1, right
Ctrl+Shift and voice "click" 0.5 (the pointer was where the gaze put it), "click on X" 0.25
(`clicks.WEIGHTS`; dune says which since 2026-11-03, `by`). The eyes' openness counts because looking down the lids close and the iris
reads higher than it is (on 400 clicks: 221 px off without it, 193 with it, each click
left out); a fix learnt before it keeps working until the next fit replaces it. With a dune from
2026-11-02 on, each click also keeps the tile it was on: its app, where in it and its size
(`ctx` in `click_samples`). Nothing uses it yet: it is the data for learning where clicks
go inside each app's window, which stays put while the strip slides. `-v`
prints each click and why it was kept or not; the fits are always in the log. A new
`calibrate` drops them.

Moving the mouse takes the pointer from the tracker: it stays where the hand leaves it
(the head does not move it either) until you look elsewhere (the gaze moves 300 px from
where it was when the hand stopped). So when it lands off, put it right with the mouse
and click: that click is where you look, and teaches it.

A `measure` run already has those points: `dune-eye-tracker calibrate --from RUN.json
--output HDMI-A-1` keeps them instead (no window; the monitor it was on).

Then `dune-eye-tracker toggle` starts it in the background, and again stops it (a key:
`"mod+i" = "dune-eye-tracker toggle"` in `[spawn]` of dune.toml). Without a
calibration it opens `calibrate`. `dune-eye-tracker run [-v] [--head-gain PX]` is the
same in the foreground.

**Landing on buttons.** With [dune-a11y](https://github.com/dune-wm/dune-a11y) installed, a jump lands on the
middle of the button, link, tab or menu item nearest where you look (within 90 px,
`--snap PX`; 0: off), from the apps' accessibility trees: at each jump `dune-a11y serve
OUTPUT` is asked what is near (it reads only that part of that window, in a few ms, and
nothing between jumps); the pointer lands on the gaze and glides over when it answers. In a
field or a list (wider or taller than 160 px) it stays where you look, inside it. The
head still moves it from there. Apps show their tree only with accessibility on
(`dune-a11y enable`, which `dune-session` runs); floating windows have none. A click by
voice ("click reply", dune-a11y) puts the pointer on what it clicked, through
`$XDG_RUNTIME_DIR/dune-eye-tracker.sock`: it stays there, as after the mouse, until you
look elsewhere, and teaches it a little (a click there, the least trusted: you looked at it
somewhere).

**Zoom for the last pixels.** dune's `zoom` key (Mod+z, or saying "zoom") magnifies the
tile under the pointer 3x around it. While it is in, the gaze on the magnified tile is
mapped back to real pixels (`unzoom`: C + (g − C)/3), the head moves the pointer a third
as fast and a jump takes a third of the distance, so the last pixels are three times
finer; no button landing meanwhile. The key again (or "click", or any click) clicks there
and goes out. A click made while zoomed is learnt where it was seen (`zoom_x`, `zoom_y`:
where the eyes were), the real point kept in its `ctx`.

How it moves: the gaze gets the pointer near (a webcam is ~3° off, ~120 px): when you
look more than 300 px away from the pointer for 0.15 s, it jumps there. Between jumps the
head moves it from where it landed: turn it a little for the last few pixels (1200 px per
unit of head turn: 3° ≈ 60 px, slow to land on a point; `--head-gain`). Gaze and head go through One Euro
filters, and the pointer glides where they take it (`--glide`, 0.1 s; 0: put there).
Turning the head to one side further than looking at the screen takes it (the eyes can
stay on the screen) steps the strip one tile that way (`dune action focus_left` /
`focus_right`: the tile is focused and shown); the pointer stays while the head is turned,
and the next tile waits for the head to come back towards the middle, then lands where
you look. How far: past the 1% most turned of your clicks (with 200 or more), else 0.35
of `head_x` from the calibration; the log says it at start, and `-v` shows `head_x` every
second and each step. Without a face, or blinking, the pointer stays. Looking off the screen (150 px
past an edge for 0.3 s, the head's turn counted: the monitor below, the keyboard) lets
go of the pointer for the mouse; looking back well on the screen takes it again. Clicks: the mouse, for now.
The camera serves one program at a time: close dune-camera first; the log is
`~/.local/state/dune-eye-tracker/run.log`.

## Measure

How well the camera knows where you look, with your glasses and your light.

```sh
uv run dune-eye-tracker measure --distance-cm 60 --width-cm 67
```

Close dune-camera first: the webcam serves one program at a time. The window goes
fullscreen on the monitor it opens on (dune is asked once it has focus;
`--no-fullscreen` not to). Space starts: follow each orange point with your eyes until
it is gone, head still, glasses on: 9 points to calibrate, 8 others to test (~40 s).

Then, for each way of guessing the gaze:

| set | from |
|---|---|
| `eyes` | the irises in their eyes |
| `head` | where the face points, and where it is |
| `both` | the two together |

it shows how far off it is on the test points, in px (and degrees with `--distance-cm`,
from your eyes to the screen, and `--width-cm`, the width of the monitor's picture):
the median, mean and worst error of a point, and the jitter (how much the guess shakes
while you look at one point). A yellow dot shows where the best one thinks you look.
Space measures again, Esc quits.

Each run (numbers only, no images) is saved in `~/.local/state/dune-eye-tracker/runs/`;
`uv run dune-eye-tracker analyze RUN.json` shows its report again.

## How

MediaPipe Face Landmarker (the model in `~/.cache/dune-eye-tracker/face_landmarker.task`)
gives 478 points of the face, irises included, and its pose; `features.py` turns them
into numbers, `model.py` fits a ridge polynomial from them to the screen (the `eyes`
set drives the pointer), `track.py` turns gaze and head into pointer moves, sent
straight on dune's socket (`pointer\tmove\tX\tY\tOUTPUT`, what `dune pointer move` sends);
`snap.py` asks `dune-a11y serve` which button a jump should land on.
[docs/click-model.md](docs/click-model.md): how a model of where clicks go is trained on
the clicks of use (the data kept, how it is checked, the stages, what was found). Its data:
every click, a line in `~/.local/state/dune-eye-tracker/clicks.jsonl`, never cut and kept
across calibrations. With each click, for a model that goes by geometry and not by app:
how the pointer got there (the jump, the gaze then even past an edge, what the head did
after it), the focused tile, and what could be clicked within 250 px of the jump
(dune-a11y: rectangles and roles, which one had the keyboard focus; no names).

## Tests

```sh
uv run pytest
```
