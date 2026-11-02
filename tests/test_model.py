import numpy as np

from dune_eye_tracker import model
from dune_eye_tracker.features import LEFT_EYE, RIGHT_EYE, eye, features
from dune_eye_tracker.measure import CALIBRATION, TEST, analyze


def fake_feats(x, y, rng, noise=0.0):
    """Features of someone looking at (x, y) in 0..1, eyes doing it all."""
    return {
        "eye_u": 0.3 * (x - 0.5) + 0.05 * (x - 0.5) ** 2 + rng.normal(0, noise),
        "eye_v": 0.15 * (y - 0.5) + rng.normal(0, noise),
        "head_x": rng.normal(0, 0.01),
        "head_y": rng.normal(0, 0.01),
        "pos_x": rng.normal(0, 0.1),
        "pos_y": rng.normal(0, 0.1),
        "pos_z": -50.0,
        "open": 0.3,
    }


def fake_run(noise):
    rng = np.random.default_rng(1)
    run = {"size": [2560, 1080]}
    for name, points in (("calibration", CALIBRATION), ("test", TEST)):
        run[name] = [
            {"point": i, "target": [x, y], "feats": fake_feats(x, y, rng, noise)}
            for i, (x, y) in enumerate(points)
            for _ in range(30)
        ]
    return run


def test_eyes_alone_are_found_when_they_say_it_all():
    models, report = analyze(fake_run(noise=0.0))
    assert models["eyes"][1].median < 5
    assert models["both"][1].median < 30
    assert "eyes" in report and "jitter" in report


def test_noise_shows_as_jitter_and_head_alone_knows_nothing():
    models, _ = analyze(fake_run(noise=0.003))
    eyes = models["eyes"][1]
    assert eyes.jitter > 10
    assert models["head"][1].median > 5 * eyes.median


def test_blinks_are_left_out():
    run = fake_run(noise=0.0)
    for s in run["test"][:30]:
        s["feats"] = s["feats"] | {"open": 0.01, "eye_u": 9.0}
    models, _ = analyze(run)
    assert models["eyes"][1].worst < 5


def test_degrees():
    # 1 cm at 57.3 cm is about 1 degree.
    assert abs(model.degrees(40, 40, 57.3) - 1.0) < 0.01


def test_iris_in_the_eye():
    pts = np.zeros((478, 2))
    for idx, x0 in ((RIGHT_EYE, 100.0), (LEFT_EYE, 200.0)):
        pts[idx["left"]] = (x0, 50)
        pts[idx["right"]] = (x0 + 40, 50)
        pts[idx["top"]] = (x0 + 20, 44)
        pts[idx["bottom"]] = (x0 + 20, 56)
        pts[idx["iris"]] = (x0 + 30, 54)  # a quarter width right, a tenth down
    u, v, o = eye(pts, RIGHT_EYE)
    assert (round(u, 3), round(v, 3), round(o, 3)) == (0.25, 0.1, 0.3)
    f = features(pts, None)
    assert round(f["eye_u"], 3) == 0.25 and f["head_x"] == 0.0


def test_the_head_guard_says_when_the_head_went():
    from dune_eye_tracker.measure import HeadGuard

    still = {"head_x": 0.0, "head_y": -0.05, "pos_x": 6.5, "pos_y": -1.9}
    guard = HeadGuard([still, still | {"head_x": 0.004}, still | {"head_x": -0.004}])
    assert guard.off(still | {"head_x": 0.02}) is None
    assert guard.off(still | {"head_x": 0.27}) == "turned"  # a corner looked at with the head
    assert guard.off(still | {"head_y": -0.12}) == "turned"
    assert guard.off(still | {"pos_x": 8.5}) == "moved"


def test_the_report_splits_the_error_and_smooths_the_jitter():
    models, report = analyze(fake_run(noise=0.003), distance_cm=60, width_cm=67)
    r = models["eyes"][1]
    assert r.smooth < r.jitter / 2  # 8 averaged: about a third
    assert max(r.x, r.y) <= r.median * 1.01 + 1
    assert "smooth 8" in report and "deg" in report


def test_head_fix_learns_what_turning_the_head_does():
    # someone keeps looking at a point while the head turns: the eyes turn the other way
    rng = np.random.default_rng(3)
    size = (2560, 1080)
    cal_rows, cal_t, cal_g = [], [], []
    for i, (x, y) in enumerate(CALIBRATION):
        for _ in range(20):
            cal_rows.append(fake_feats(x, y, rng, 0.001))
            cal_t.append((x * size[0], y * size[1]))
            cal_g.append(i)
    eyes = model.fit(cal_rows, np.array(cal_t), np.array(cal_g), "eyes")
    ref = {"head_x": 0.0, "head_y": 0.0, "pos_x": 0.0, "pos_y": 0.0}
    rows, targets, groups = [], [], []
    for i, (x, y) in enumerate([(0.5, 0.5), (0.2, 0.25), (0.8, 0.25), (0.2, 0.75), (0.8, 0.75)]):
        for hx in np.linspace(-0.08, 0.08, 9):
            for hy in np.linspace(-0.05, 0.05, 5):
                f = fake_feats(x, y, rng, 0.001)
                f |= {"head_x": hx, "head_y": hy, "eye_u": f["eye_u"] + 0.6 * hx, "eye_v": f["eye_v"] - 0.4 * hy}
                rows.append(f)
                targets.append((x * size[0], y * size[1]))
                groups.append(i)
    fix, alone, fixed = model.fit_head(eyes, rows, np.array(targets), np.array(groups), ref)
    assert alone > 150  # the eyes alone are thrown off by the head
    assert fixed < alone / 3
    back = model.HeadFix.from_json(fix.to_json())
    assert np.allclose(back.correct(np.zeros((1, 2)), rows[:1]), fix.correct(np.zeros((1, 2)), rows[:1]))


def test_a_fix_saved_before_openness_still_loads_and_corrects():
    old = {"ref": [0.0, 0.0, 0.0, 0.0], "bias": [10.0, -5.0], "weights": [[100.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]}
    fix = model.HeadFix.from_json(old)
    assert fix.names == model.HEAD_COLUMNS
    row = {"head_x": 0.1, "head_y": 0.0, "pos_x": 0.0, "pos_y": 0.0, "open": 0.3}
    assert np.allclose(fix.correct(np.zeros((1, 2)), [row]), [[20.0, -5.0]])


def test_the_fix_learns_what_closing_the_eyes_does():
    # looking down the lids close and the iris reads higher: the openness tells it
    rng = np.random.default_rng(5)
    size = (2560, 1080)
    cal_rows, cal_t, cal_g = [], [], []
    for i, (x, y) in enumerate(CALIBRATION):
        for _ in range(20):
            cal_rows.append(fake_feats(x, y, rng, 0.001) | {"open": 0.27})
            cal_t.append((x * size[0], y * size[1]))
            cal_g.append(i)
    eyes = model.fit(cal_rows, np.array(cal_t), np.array(cal_g), "eyes")
    ref = dict.fromkeys(model.HEAD_COLUMNS, 0.0)
    rows, targets, groups = [], [], []
    for i, (x, y) in enumerate([(0.5, 0.5), (0.2, 0.25), (0.8, 0.25), (0.2, 0.75), (0.8, 0.75), (0.5, 0.9)]):
        for op in np.linspace(0.20, 0.34, 8):
            f = fake_feats(x, y, rng, 0.001)
            f |= {"open": op, "eye_v": f["eye_v"] - 0.8 * (0.27 - op)}  # less open: reads higher
            rows.append(f)
            targets.append((x * size[0], y * size[1]))
            groups.append(i)
    fix, alone, fixed = model.fit_head(eyes, rows, np.array(targets), np.array(groups), ref, size)
    assert fix.names == model.FIX_COLUMNS
    assert fixed < alone / 2
    back = model.HeadFix.from_json(fix.to_json())
    assert back.names == fix.names
    assert np.allclose(back.correct(np.zeros((2, 2)), rows[:2]), fix.correct(np.zeros((2, 2)), rows[:2]))
