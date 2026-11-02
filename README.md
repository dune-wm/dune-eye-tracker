# dune-eye-tracker

The mouse pointer in [dune](../dune), moved by where you look, with a webcam.

## Drive the pointer

```sh
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
hears them from `dune events pointer` (the mouse's, and dune-voice's "click"), keeps the
camera's samples from just before a click on the calibrated monitor when the gaze was
steady, wherever it guessed you looked (it trusts you look at the pointer), and every 5 clicks (the first after 10) fits a fix (head
and gaze, `clicks.py`) on them, used when it does no worse on the last 5 clicks. `-v`
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

How it moves: the gaze gets the pointer near (a webcam is ~3° off, ~120 px): when you
look more than 300 px away from the pointer for 0.15 s, it jumps there. Between jumps the
head moves it from where it landed: turn it a little for the last few pixels (2000 px per
unit of head turn: 3° ≈ 100 px, `--head-gain`). Gaze and head go through One Euro
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
straight on dune's socket (`pointer\tmove\tX\tY\tOUTPUT`, what `dune pointer move` sends).

## Tests

```sh
uv run pytest
```
