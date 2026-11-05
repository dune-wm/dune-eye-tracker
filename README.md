# dune-eye-tracker

The mouse pointer in [dune](https://github.com/dune-wm/dune), moved by where you look, with a webcam.

It all runs on this computer: the camera's images are read and dropped in memory, never
saved nor sent; only numbers are kept (the calibration, clicks, games, recordings; with the
clicks, the roles of what could be clicked near them), in `~/.local/state/dune-eye-tracker`.
While `run` goes it can share the camera with other programs on this computer through a
virtual one (below). The face model (MediaPipe's `face_landmarker.task`) is downloaded
once, by hand.

## Drive the pointer

```sh
git clone https://github.com/dune-wm/dune-eye-tracker && cd dune-eye-tracker
uv tool install --editable .        # dune-eye-tracker in ~/.local/bin
dune-eye-tracker calibrate          # once: 28 points, about a minute
```

`calibrate` goes fullscreen on the monitor the pointer is on: that is the monitor it
drives. Look at each orange point until it is gone, sitting as you usually do (the head
may move). It is saved in `~/.local/state/dune-eye-tracker/gaze.npz` (numbers: the face's
landmarks, its pose, the iris found; no images); a calibration it replaces is kept beside
it with its time. A recording's calibration points can be taken instead, no window:
`dune-eye-tracker calibrate --from REC.npz --output HDMI-A-1` (`record`, `live`'s C, or
dune-gaze's).

Then `dune-eye-tracker toggle` starts it in the background, and again stops it (a key:
`"mod+i" = "dune-eye-tracker toggle"` in `[spawn]` of dune.toml). Without a calibration it
opens `calibrate`. `dune-eye-tracker run [-v]` is the same in the foreground; the log is
`~/.local/state/dune-eye-tracker/run.log`.

**Your clicks keep it calibrated**, with nothing to sit through again: look at the pointer
when you click. Every click on its monitor (the mouse's, right Ctrl+Shift, dune-voice's
"click", "click on X" through dune-a11y) with the gaze steady just before it adds those
frames as a calibration point, wherever it guessed you looked. Sitting differently moves
the gaze (once ~130 px down): the last 9 mouse clicks give a vertical shift taken off every
guess, from the 5th on. Every 5 clicks a new fit on the calibration and the last 300 clicks,
in a thread, is used when it does no worse on the last 50 (half of them while there are
fewer; up to 15% worse still counts, as the fit in use has seen most of them). Not all
clicks count the same: the mouse's and the voice's ("click", "click on X") 1, right
Ctrl+Shift 0.5 (the pointer was where the gaze put it) (`clicks.WEIGHTS`). Replaying two real
sessions as clicks, the miss went from 187 to 126 px where the posture had moved. The log
says each fit and each shift; `-v` each click and what came of it.

Moving the mouse takes the pointer from the tracker: it stays where the hand leaves it
(the head does not move it either) until you look elsewhere (the gaze moves 300 px from
where it was when the hand stopped). So when it lands off, put it right with the mouse
and click: that click is where you looked, and teaches it most.

**How it moves.** The pointer goes only where you fix your eyes (a webcam is ~3° off,
~120 px). The eyes move all the time, reading, glancing, and a pointer glued to them is
everywhere: so it stays put while the gaze wanders, and once the gaze holds still (within
40 px for 0.3 s) more than 120 px from where the pointer last landed, it glides there in
one move: a landing (a "jump" in the trace and the clicks' log), where a button is looked
for. The head does not move it: the gaze already puts the eyes in the head (turned far to
a side it steps a tile, below). The gaze goes through a One Euro filter (steady, beta
0.004), and the pointer glides (`--glide`, 0.1 s; 0: put there). The gaze (and button landings)
keeps the pointer off what the tiles leave at the screen's edges, from dune's `strip`
layout: the outer `gap`, a bar's band (at most 64 px of an edge; a tile cut by the edge
leaves nothing); the mouse still goes anywhere. Without a face, or blinking, the pointer stays.
Looking off the screen (150 px past an edge for 0.3 s: the monitor below, the keyboard)
lets go of the pointer for the mouse; looking back well on the screen takes it again.

**Why it moved.** `run` writes each landing of the pointer to
`~/.local/state/dune-eye-tracker/trace.jsonl` (how: the gaze, back from a head step; from
where it was to where the eyes held, how far, how long since the last one), and what else
moved it or the gaze under it (a snap onto a button, the clicks' shift, a new fit, the
zoom, a put). A jump that goes back near where the pointer was before the last one (within
150 px, 1.5 s after a jump of 300 px or more) is a **bounce**: it is written with the last
~3 s of frames (the gaze guessed and filtered, the head, the pointer and its speed, how open
the eyes were; blinks and frames without the iris too), and `run.log` says so.
`dune-eye-tracker trace` sums it up: the jumps, the bounces, and for each how much the raw
gaze wandered and how many frames the eyes were nearly shut. `--no-trace`: not written.

**Tiles with the head.** Turning the head to one side further than looking at the screen
takes it (the eyes can stay on the screen) steps the strip one tile that way (`dune action
focus_left` / `focus_right`: the tile is focused and shown); the pointer stays while the
head is turned, and the next tile waits for the head to come back towards the middle, then
lands where you look. How far: 0.12 of `head_x` past the 1% most turned of your clicks
(with 200 or more), else 0.35 from the calibration, held there 0.25 s (reaching for a side
edge turns the head nearly as far, for a moment); the log says it at start, and `-v` shows
`head_x` every second and each step.

**Landing on buttons.** With [dune-a11y](https://github.com/dune-wm/dune-a11y) installed, a jump lands on the
middle of the button, link, tab or menu item nearest where you look (within 45 px,
`--snap PX`; 0: off), from the apps' accessibility trees: at each jump `dune-a11y serve
OUTPUT` is asked what is near (it reads only that part of that window, in a few ms, and
nothing between them); the pointer lands on the gaze and glides over when it answers. In a
field or a list (wider or taller than 160 px) it stays where you look, inside it. The
gaze's wobble does not move it from there. Apps show their tree only with accessibility on
(`dune-a11y enable`, which `dune-session` runs); floating windows have none. A click by
voice ("click reply", dune-a11y) puts the pointer on what it clicked, through
`$XDG_RUNTIME_DIR/dune-eye-tracker.sock`: it stays there, as after the mouse, until you
look elsewhere, and teaches it a little (the least trusted click: you looked at it
somewhere).

**Zoom for the last pixels.** dune's `zoom` key (Mod+z, or saying "zoom") magnifies the
tile under the pointer 3x around it. While it is in, the gaze on the magnified tile is
mapped back to real pixels (`unzoom`: C + (g − C)/3), and a fixation moves it from a third as
far (and holds within a third as much), so the last pixels are three times finer; no button landing meanwhile. The key again (or "click", or any click) clicks there
and goes out. A click made while zoomed is learnt where it was seen (`zoom_x`, `zoom_y`:
where the eyes were), the real point kept in its `ctx`.

## Share the camera

The webcam serves one program at a time. `run` keeps it; when another program needs it (a
call, dune-camera), `dune-eye-tracker share` makes `run` write every frame to a virtual
camera, `/dev/video10` ("dune camera"), that they use as a webcam while the pointer goes on,
and again stops it (a key: `"mod+shift+i" = "dune-eye-tracker share"` in `[spawn]` of
dune.toml; a notification says which). It starts off (`run --share`: on): copying 1080p
frames costs CPU for nothing the rest of the day. Without the device it says so and tracks
as ever. With `run` off the webcam is free and nothing needs sharing. Once, as root (v4l2loopback-dkms needs the running kernel's headers):

```sh
sudo pacman -S linux72-headers              # the headers of your kernel (uname -r); dkms builds the module
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback devices=1 video_nr=10 card_label="dune camera" exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf
sudo modprobe v4l2loopback                  # now; at boot from then on
ls /dev/video10
```

`exclusive_caps=1` lets browsers see it (as a capture device once `run` writes to it: share
before the call looks for cameras); dune's `dune-camera` (Mod+c) needs no `share`: with `run`
going it turns sharing on by itself and off when its window closes (unless a call reads it
then); with `run` off it shows the webcam.
Then restart `run` (`dune-eye-tracker toggle` twice). Frames go only to that device, on
this computer; nothing is saved. The commands below that open the camera themselves
(`calibrate`, `game`, `record`, `live`) need `run` off.

## The circles game

How far off the tracker is in use, on targets you did not choose, against the click log.
The goal: a median miss of 100–120 px.

```sh
dune-eye-tracker toggle   # stop `run`: the game needs the camera
dune-eye-tracker game
```

It needs a calibration and goes fullscreen on its monitor. Two rounds (`--only measure`
or `--only play` for one), 4 min at most:

- **measure** (the tracker itself): 24 points one at a time, a fixed 6 x 4 grid with the
  edges in, none a calibration point, the same every game. No pointer is shown: seeing
  it you would look off to put it right, and that is what would be measured. Look at
  each until it is gone; the gaze is taken 0.6..1.5 s after it shows. Sit as you usually
  do: the head may move.
- **play** (whether it can be used): circles of radius 30 to 180 px, one or two at a
  time, big and small in turn, spread evenly over the quadrants. No click: bring the
  yellow pointer into one with your eyes and keep it there 0.5 s; after 6 s it is missed.
  The pointer moves as in `run` (the gaze, its clicks' shift, filters, to each fixation),
  without the mouse, dune-a11y's snap or the head's tile steps. 28
  circles (`--rounds N` of each radius, 4 by default). Its misses are not the tracker's: a
  circle taken was reached, whatever the gaze.

Each point and circle is a line in `~/.local/state/dune-eye-tracker/game.jsonl` (`round`,
`seen`, `guess`, `miss`, `spread`, `feats`, `quadrant`, `model`: the calibration, its
clicks and the fit played with, `cond`; points also `raw_miss`, without the clicks' shift;
circles `radius`, `hit`, `took`, `with`, `settled`, `pointer`, `jump`). It does not teach
the gaze. The measure round is also saved as a recording, `games/<game>.npz`, with the
calibration it was played with: `dune-eye-tracker eval games/*.npz` guesses it again with
every model (one game, many ideas, no new noise).

The end shows the tracker's median miss with a 95% interval (the goal is said reached
only when the interval is under it), per quadrant (by where the point was, and by where
the guess fell: all the tracker knows in use; how the guess leans; the click log there),
what was taken per radius, and the conditions (distance, how much the head moved and
turned, fps, hours since calibrating). `game --report` pools every game of the current
gaze.

## The lab

Where the gaze was worked out (in dune-gaze, 2026-11-03..04; its
recordings in `~/.local/state/dune-gaze/recordings` are read too). `run` off for those that
open the camera:

```sh
dune-eye-tracker record    # ~2.5 min: 28 calibration points, 5 with the head turning, 24 to test
dune-eye-tracker eval      # every model on every recording, within and across them
dune-eye-tracker check     # does the face's pose hold (the geometry's premise)
dune-eye-tracker live      # the gaze as you look: C calibrates here, R adjusts (9 points), T tests a point
```

Recordings are numbers only, in `~/.local/state/dune-eye-tracker/recordings/`, unless
`--keep-eyes` (the two grey eye crops of every frame, ~160x100: images of your eyes, so a
better iris finder can be tried on old recordings). `eval` fits each model (`models.models`:
the old 2D eyes model, the geometric one with each iris, an eyes-in-head ridge) on a
recording's calibration points and gives the median miss on its grid (95% interval), on
the head-moving points, per quadrant, and across recordings (another posture). `live`
saves each session as a recording (`<time>-live.npz`).

What it found: the old model put the iris in the 2D image, and turning the head moves that
too (124 px with the head still, 360-500 free); no linear fix on it got under ~330. Putting
the eyes in the head in 3D (MediaPipe's canonical face and pose), finding the iris in the
full 1080p frame (its dark centroid: MediaPipe works on a ~256 px face crop, where one
degree of eye turn is a third of a pixel), a quadratic map from 28 points with the edges
in, and guesses kept on the screen: 114-124 px live. A saved calibration drifts with
posture (a vertical shift), which the clicks now take off.

## How

`camera.py` reads the webcam at 1920x1080 and runs MediaPipe Face Landmarker (the model in
`~/.cache/dune-eye-tracker/face_landmarker.task`: 478 points of the face and its pose) on
every frame; `iris.py` finds each iris again in the full frame. `face.py` puts the eyes in
the head (the canonical face read from the model's bundle) and the visual axes on the
camera's plane; `models.py` maps that to the screen (`Geometric`); `gaze.py` keeps the
calibration and fits it again on the clicks. `track.py` turns gaze and head into pointer
moves, sent straight on dune's socket (`pointer\tmove\tX\tY\tOUTPUT`, what `dune pointer
move` sends); `snap.py` asks `dune-a11y serve` which button a jump should land on;
`clicks.py` hears dune's clicks and logs them.
[docs/click-model.md](docs/click-model.md): how a model of where clicks go is trained on
the clicks of use. Its data: every click, a line in
`~/.local/state/dune-eye-tracker/clicks.jsonl`, never cut and kept across calibrations.
With each click, for a model that goes by geometry and not by app: how the pointer got
there (the jump, the gaze then even past an edge, what the head did after it), the focused
tile, and what could be clicked within 250 px of the jump (dune-a11y: rectangles and roles,
which one had the keyboard focus; no names).

## Tests

```sh
uv run --group dev python -m pytest
```

Synthetic eyes (a known head and gaze projected; the geometry must give them back), the
pointer, the clicks, the game.
