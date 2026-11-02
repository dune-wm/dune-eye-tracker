# A model of where clicks go

A living document: how to train, check and use a small model that tells where the pointer
should go, learnt from the clicks of use, on top of the gaze. Started 2026-11-02; each
stage adds what it found to the log at the end.

## The idea

The gaze model gets the pointer near what you look at, and no nearer: on a webcam the best
published models miss by about 4° (L2CS-Net, 3.92° on MPIIGaze), and ours misses by ~3.4°
(124 px) on a calibration test and ~190 px in use. What is left is not a bias to subtract:
two clicks on the same spot are guessed ~195 px apart, the mean miss is ~0 on every part of
the screen, and the miss of one click says nothing of the next (see the log). So the gaze
is not pushed further. A second model learns something else: **where you click**, in each
app, so a landing near a place you click often can go there.

On the screen that does not work: the strip slides and pages scroll, so a button changes
pixel, and with ~200 px of error many past spots are equally near (tried: no gain, worse
than the gaze on half the clicks). Inside its window a button stays put. dune says, for
every click, which window it was on and where in it; that is what the model learns from.

## The data

**The model's data is `~/.local/state/dune-eye-tracker/clicks.jsonl`**: one line per click
on the calibrated monitor (`ClickLearner.log`), never cut and kept across calibrations, so
it grows for as long as more clicks still help (the learning curve is what says when to
stop caring, not a cap; ~400 bytes a click, 100 000 clicks ~40 MB).

| Field | What |
|---|---|
| `time` | when (local), for recency and for splitting in order |
| `calibration` | the calibration in use (its `time`): guesses of different calibrations differ |
| `output`, `size` | the monitor and its size |
| `seen` | where the click was seen, px (the magnified point while zoomed) |
| `guess` | the gaze guess then (the median of the frames before it, with the click fix) |
| `spread` | how much the guess moved in those frames, px: above `STEADY` (100) the click fix does not learn from it; the model may |
| `feats` | the eyes' and head's numbers (median of the frames): a guess can be made again with another calibration |
| `ctx` | the tile, as below |

The click fix has its own, bounded data (it is fitted on every frame, and only means
something for one calibration): each click it keeps goes in
`~/.local/state/dune-eye-tracker/calibration.json`,
`click_samples`: one row per camera frame of the 0.35–0.05 s before the click (the eyes
there, the hand not yet), at most `KEEP` (1000) clicks, the last ones.

| Field | What |
|---|---|
| `point` | the click's id (every frame of one click has it) |
| `target` | where the click was **seen**, 0..1 of the monitor (x / width, y / height). While dune's zoom was in, the magnified point: the eyes were there |
| `feats` | the frame: `eye_u`, `eye_v`, `head_x`, `head_y`, `pos_x`, `pos_y`, `pos_z`, `open` (see `features.py`) |
| `ctx` | from dune (a dune from 2026-11-02 on), when the click was on a tile: `tile`, `app_id`, `tile_x`, `tile_y` (from the tile's corner), `tile_w`, `tile_h`; and while zoomed `real_x`, `real_y` (where the click went) and `zoom_scale` |

The gaze guess for a frame is `eyes.predict(feats)` corrected by the click fix (`track.
load_calibration`, `track.head_fix`); a click's guess is the median of its frames'.

Not kept yet, worth adding when a stage needs it: the time of the click (for recency), the
window's title (a page, not only an app), and whether the pointer was put there by the
gaze, the hand, a voice or the zoom (a click after the hand moved the pointer says the
gaze was not enough).

## Checking a model honestly

The clicks are a time series: a model may only learn from clicks before the one it is
checked on, as `run` would.

- **Split**: walk the clicks in order; for click *i*, fit (or update) on clicks < *i*,
  check on *i*. For slow fits, refit every 25 clicks.
- **Measures**, against the gaze guess alone on the same clicks:
  - median miss (px) and per region (thirds of the screen);
  - hits: share within 40 px (a button is ~30 px tall);
  - **worse than the gaze**: share of clicks the model moves further away. A model that
    wins on average but loses on half of the clicks feels worse than none.
- **Per app**: an app with few clicks must fall back to the gaze, not to another app's
  habits.
- Report every attempt in the log below, kept or not.

## Stages

1. **Collect** (now). dune 2026-11-02 (needs a new session) gives `ctx`; every click goes in
   `clicks.jsonl`. Use the tracker as usual; nothing changes in how it behaves. Start
   checking at ~500 clicks with `ctx`; keep collecting while the learning curve (the
   model's miss on later clicks against how many it was fitted on) still goes down, per
   app: a much-used app settles long before a rare one.
2. **A density per app, in window coordinates.** For each `app_id`: where in the window
   clicks fall, normalised (`tile_x / tile_w`, `tile_y / tile_h`) and also in pixels from
   each corner (a toolbar keeps its place from the top-left, a send button from the
   bottom-right, whatever the size). A kernel density (σ ~15 px), recent clicks weighted
   more. Cheap, explainable, a few kB.
3. **Combine with the gaze** (Bayes): for a landing at gaze g in window w of app a,
   score each candidate spot s (the density's peaks inside w) by
   `P(g | s) · P(s | a)`, with `P(g | s)` a Gaussian of the measured miss (σ ~200 px, per
   axis from the data). Land on the best spot only when it is clearly better than the gaze
   itself (posterior ratio, tuned on the checks above), else on the gaze.
4. **A small learned model**, only if 2–3 leave a gap: gradient boosting or a tiny MLP
   (numpy, no GPU) from (gaze guess in window coordinates, `head_*`, `open`, app, window
   size, the density's nearest peaks) to the click's window position. Trained in the
   background like the click fix, checked as above before it is used.
5. **Use it.** At a jump: gaze → (if the model is confident) its spot; then `dune-a11y
   serve` lands on the button there; the zoom (Mod+z) stays for what neither got right.
   Every click keeps teaching both.

## Tools (decided 2026-11-02)

No LLM and no deep network: the data is small and tabular (a few numbers per click, a few
thousand clicks), where a big model only learns the clicks by heart.

- **Stages 2–3: numpy only** (already a dependency): the per-app density and the Bayes
  step are counting, smoothing and a product, a few lines whose every decision can be
  read.
- **Stage 4: scikit-learn.** `LogisticRegression` (or `SGDClassifier.partial_fit`, to
  update click by click) ranking the candidates of a landing (the buttons near the gaze
  from `dune-a11y serve`, the app's frequent spots) on their distance to the gaze, how
  often and how lately they were clicked, their size and role; `HistGradientBoosting
  Classifier` only if the logistic falls short; `TimeSeriesSplit` for the checks in order;
  `KernelDensity` to compare with the hand-made density.
- **Analysis first**: `uv add --group analysis scikit-learn` when stage 4 starts, not a
  dependency of `run` until a model has passed its checks and is used live.
- **A model is saved as JSON, not pickle**: a logistic regression is a few dozen numbers,
  safe to load and readable, and run with numpy in the pointer's loop without importing
  scikit-learn. Trees would need scikit-learn at run time: one more reason to try them
  only when needed.
- Not LightGBM or XGBoost: nothing to gain at a few thousand clicks, native libraries more.

## Guards

- Nothing leaves the computer; the data is the user's own clicks.
- A model is used only when its checks beat the gaze, the same rule as the click fix
  (`clicks.py`: no worse on the last 50 clicks, with its margin).
- Off by a flag, like `--snap 0`.

## Log

- **2026-11-02**, 400 clicks without `ctx` (tools: throwaway scripts; to be kept in
  `tools/` when a stage needs them again):
  - gaze alone 386 px median miss; with the click fix 219 px; + eye openness 193 px.
  - a remap of the guess by screen position (linear / quadratic / cubic, each click left
    out): 218 / 216 / 206 px. No slow drift: correcting by the previous k clicks' mean
    miss gives nothing; lag correlation ~0 after 5 clicks.
  - two clicks within 60 px are guessed 195 px apart (frame jitter is ~25 px): the miss is
    per click.
  - 49% of clicks fall within 40 px of an earlier one, but a prior on screen positions
    does not help: nearest earlier click 210–219 px; weighted by frequency 216–243 px,
    hits up to 9% (from 2%), worse than the gaze on 46–59% of clicks. Hence window
    coordinates (`ctx`).
  - learning curve of the click fix (fit on the last N before 100 unseen clicks): 260 /
    220 / 194 px for N = 100 / 200 / 300, still going down; `KEEP` raised to 1000.
