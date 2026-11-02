import json

import numpy as np

from dune_eye_tracker import clicks, model
from dune_eye_tracker.measure import CALIBRATION
from test_model import fake_feats

SIZE = (2560, 1080)


def calibration(rng):
    samples = [
        {"point": i, "target": [x, y], "feats": fake_feats(x, y, rng, 0.001)}
        for i, (x, y) in enumerate(CALIBRATION)
        for _ in range(20)
    ]
    data = {"size": list(SIZE), "output": "HDMI-A-1", "head": dict.fromkeys(model.HEAD_COLUMNS, 0.0), "calibration": samples}
    rows = [s["feats"] for s in samples]
    targets = np.array([[s["target"][0] * SIZE[0], s["target"][1] * SIZE[1]] for s in samples])
    eyes = model.fit(rows, targets, np.array([s["point"] for s in samples]), "eyes")
    return data, eyes


def look(learner, t0, x, y, rng, drift=(0.0, 0.0), head=0.0):
    """Frames of someone looking at (x, y) px for 0.5 s; the eyes model off by `drift`
    (0..1 of the screen) and the head turned by `head`."""
    for i in range(15):
        f = fake_feats(x / SIZE[0] + drift[0], y / SIZE[1] + drift[1], rng, 0.001)
        f |= {"head_x": head, "eye_u": f["eye_u"] + 0.6 * head}
        learner.seen(t0 + i / 30, f)
    return t0 + 0.5


def test_a_click_is_trusted_however_far_the_guess(tmp_path):
    rng = np.random.default_rng(4)
    data, eyes = calibration(rng)
    learner = clicks.ClickLearner(data, eyes, None, tmp_path / "c.json")
    t = look(learner, 0.0, 1000, 500, rng)
    assert learner.click(t, 1010, 505, "HDMI-A-1").startswith("kept")
    # the guess 1400 px off (not calibrated yet): you were looking at the pointer all the same
    t = look(learner, 1.0, 1000, 500, rng)
    assert learner.click(t, 2400, 900, "HDMI-A-1").startswith("kept (14")
    assert learner.click(t, 1000, 500, "HDMI-A-2") == "other monitor"
    assert learner.click(50.0, 1000, 500, "HDMI-A-1") == "no face before it"
    assert learner.kept == 2


def test_clicks_learn_what_the_calibration_got_wrong(tmp_path):
    # the eyes model is off (as if you sat differently) and the head turns: clicks fix it
    rng = np.random.default_rng(5)
    data, eyes = calibration(rng)
    path = tmp_path / "c.json"
    learner = clicks.ClickLearner(data, eyes, None, path)
    notes, t = [], 0.0
    for i in range(40):
        x, y = rng.uniform(200, 2360), rng.uniform(150, 930)
        head = rng.uniform(-0.05, 0.05)
        t = look(learner, t, x, y, rng, drift=(0.06, -0.05), head=head)
        assert learner.click(t, int(x), int(y), "HDMI-A-1").startswith("kept"), i
        if note := learner.refit():
            notes.append(note)
        t += 1.0
    assert len(notes) == 7 and any(n.endswith("used") for n in notes)  # at 10, 15, ... 40 clicks
    rows, targets, groups = learner.arrays(learner.samples[-150:])
    alone = model.error(eyes.predict(rows), targets, groups)
    assert alone > 120  # the calibration is that far off
    assert data["click_fix"]["after"] < 40  # the clicks put it right
    saved = json.loads(path.read_text())
    assert len({s["point"] for s in saved["click_samples"]}) == 40
    assert model.HeadFix.from_json(saved["click_fix"]).gaze_mid is not None


def test_a_fit_in_the_background_lands_later_and_the_clicks_go_on(tmp_path):
    rng = np.random.default_rng(6)
    data, eyes = calibration(rng)
    learner = clicks.ClickLearner(data, eyes, None, tmp_path / "c.json")
    t, notes = 0.0, []
    for i in range(12):
        x, y = rng.uniform(200, 2360), rng.uniform(150, 930)
        t = look(learner, t, x, y, rng, drift=(0.06, -0.05))
        learner.click(t, int(x), int(y), "HDMI-A-1")
        if note := learner.refit_background():
            notes.append(note)
        t += 1.0
    assert learner.job is not None  # due at 10 clicks: started, not waited for
    learner.job.join(timeout=10)
    note = learner.refit_background()  # done: put in use
    assert note.startswith("10 clicks: on the last 5")  # the clicks as they were when it started
    assert learner.job is None and learner.fix is not None
    assert (tmp_path / "c.json").exists()


def test_the_tile_a_click_was_on_is_kept_with_it(tmp_path):
    rng = np.random.default_rng(7)
    data, eyes = calibration(rng)
    path = tmp_path / "c.json"
    learner = clicks.ClickLearner(data, eyes, None, path)
    line = {"event": "pointer", "change": "click", "button": "left", "x": 812, "y": 440, "output": "HDMI-A-1",
            "tile": 3, "app_id": "vivaldi-stable", "tile_x": 200, "tile_y": 96, "tile_w": 1268, "tile_h": 1064}
    ctx = clicks.context(line)
    assert ctx == {"tile": 3, "app_id": "vivaldi-stable", "tile_x": 200, "tile_y": 96, "tile_w": 1268, "tile_h": 1064}
    assert clicks.context({"event": "pointer", "change": "click", "x": 1, "y": 2, "output": "M"}) is None  # an older dune
    t = look(learner, 0.0, 812, 440, rng)
    assert learner.click(t, 812, 440, "HDMI-A-1", ctx).startswith("kept")
    t = look(learner, 1.0, 900, 500, rng)
    assert learner.click(t, 900, 500, "HDMI-A-1").startswith("kept")  # off a tile: none
    learner.save()
    saved = json.loads(path.read_text())["click_samples"]
    assert all(s["ctx"] == ctx for s in saved if s["point"] == saved[0]["point"])
    assert not any("ctx" in s for s in saved if s["point"] != saved[0]["point"])
