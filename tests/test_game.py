import math

import numpy as np

from dune_eye_tracker import game
from dune_eye_tracker.record import CALIBRATION

SIZE = (2560, 1080)


def test_points_a_grid_of_six_a_quadrant_none_a_calibration_point():
    assert not any(p in CALIBRATION for p in game.POINTS)
    qs = [game.quadrant(x * SIZE[0], y * SIZE[1], SIZE) for x, y in game.POINTS]
    assert [qs.count(q) for q in range(4)] == [6, 6, 6, 6]


def test_point_order_every_point_once_jumping():
    o = game.point_order(np.random.default_rng(2), SIZE)
    assert sorted(o) == list(range(len(game.POINTS)))
    far = [math.hypot((game.POINTS[a][0] - game.POINTS[b][0]) * SIZE[0], (game.POINTS[a][1] - game.POINTS[b][1]) * SIZE[1]) >= game.APART for a, b in zip(o, o[1:])]
    assert sum(far) >= len(far) - 2  # only the last few may be stuck near


def test_order_every_radius_big_and_small_in_turn():
    o = game.order(np.random.default_rng(3))
    assert sorted(o) == sorted(float(r) for r in game.RADII for _ in range(game.ROUNDS))
    mid = sorted(game.RADII)[len(game.RADII) // 2]
    pairs = [(a, b) for a, b in zip(o, o[1:]) if mid not in (a, b)]
    assert all((a > mid) != (b > mid) for a, b in pairs)


def test_place_fills_the_quadrants_evenly_on_the_screen_and_apart():
    rng = np.random.default_rng(5)
    counts, last = [0, 0, 0, 0], None
    for r in game.order(rng):
        x, y, q = game.place(SIZE, r, rng, counts, [], last)
        assert game.quadrant(x, y, SIZE) == q
        assert r + game.EDGE <= x <= SIZE[0] - r - game.EDGE
        assert r + game.EDGE <= y <= SIZE[1] - r - game.EDGE
        if last is not None:
            assert math.hypot(x - last[0], y - last[1]) >= game.APART
        counts[q] += 1
        last = (x, y)
    assert counts == [7, 7, 7, 7]


def test_place_a_second_circle_in_another_quadrant_clear_of_the_first():
    rng = np.random.default_rng(7)
    up = [(600.0, 300.0, 180.0)]
    for _ in range(50):
        x, y, q = game.place(SIZE, 180.0, rng, [0, 0, 0, 0], up)
        assert q != 0
        assert math.hypot(x - 600, y - 300) >= 360 + game.GAP


def test_a_frame_counts_for_the_circle_nearest_the_gaze():
    up = [(500.0, 500.0, 30.0), (2000.0, 500.0, 180.0)]
    assert game.nearest((700, 480), up) == 0
    assert game.nearest((1500, 480), up) == 1


def frames(guess, raw=None, start=0.0, end=2.0, step=1 / 30):
    feats = {"head_x": 0.1, "open": 0.3}
    n = int(round((end - start) / step))
    return [{"t": start + i * step, "guess": list(guess), "raw": list(raw or guess), "feats": feats} for i in range(n + 1)]


def test_point_line_from_the_window_after_it_shows():
    f = frames((1000, 500), start=0.0, end=0.5) + frames((1030, 540), (1100, 500), start=0.6, end=1.5)
    line = game.point_line(f, (1000, 500), 0.0)
    assert line["guess"] == [1030, 540] and line["miss"] == 50 and line["raw_miss"] == 100
    assert line["spread"] == 0


def test_trial_line_is_shaped_like_a_click():
    line = game.trial_line(frames((1030, 540)), (1000, 500, 50), 0.0, 2.0, True, at=(1010, 540))
    assert line["seen"] == [1000, 500] and line["radius"] == 50 and line["hit"]
    assert line["guess"] == [1030, 540] and line["spread"] == 0
    assert line["miss"] == 50 and line["settled"] == 50
    assert math.isclose(line["pointer"], math.hypot(10, 40), abs_tol=0.1)


def test_trial_line_of_a_circle_not_looked_at():
    line = game.trial_line([], (500, 500, 30), 0.0, 6.0, False)
    assert "guess" not in line and "pointer" not in line and not line["hit"]


def test_ci_holds_the_median_and_narrows_with_more():
    rng = np.random.default_rng(0)
    few, many = rng.normal(150, 50, 10), rng.normal(150, 50, 200)
    lo, hi = game.ci(few)
    assert lo <= np.median(few) <= hi
    lo2, hi2 = game.ci(many)
    assert hi2 - lo2 < hi - lo
    assert game.ci([1, 2]) is None


def test_goal_said_only_when_the_interval_says_it():
    assert game.goal_said([80, 90, 95, 100, 85, 90, 92]) == "reached"
    assert game.goal_said([200, 210, 190, 205, 220, 198]).startswith("not yet")
    assert game.goal_said([60, 200, 90, 180, 110, 150]).startswith("cannot tell")


def test_conditions():
    feats = [{"pos_z": -40.0, "pos_x": x, "pos_y": 0.0, "head_x": 0.0, "head_y": 0.0} for x in (-1.0, 1.0)]
    c = game.conditions(feats, [27.0, 25.0], 2.0)
    assert c == {"distance": 40.0, "moved": 1.0, "turned": 0.0, "fps": 26.0, "hours": 2.0}


def test_report_measure_per_quadrant_play_and_conditions():
    s = list(SIZE)
    pts = [{"round": "measure", "seen": [100, 100], "guess": [100 + m, 100], "miss": m, "raw_miss": m + 20, "size": s, "cond": {"distance": 40.0}} for m in (90.0, 100.0, 110.0)]
    circles = [{"round": "play", "radius": 30, "hit": True, "took": 2.0, "size": s}, {"round": "play", "radius": 30, "hit": False, "took": 6.0, "size": s}]
    clicks = [{"seen": [0, 0], "guess": [180, 0], "size": s}, {"seen": [0, 0], "guess": [0, 200], "size": s}]
    text = "\n".join(game.report(pts + circles, clicks))
    assert "3 points measured, 2 circles played" in text
    assert "miss 100 (" in text and "without the clicks' shift 120 (" in text
    assert "goal 100-120 px: reached" in text
    top_left = next(r for r in text.splitlines() if r.strip().startswith("top left")).split()
    assert top_left[2] == "3" and top_left[3] == "100" and top_left[-1] == "190"
    assert "play: 1/2 circles taken" in text
    assert "conditions: 40 cm away" in text
    assert "the click log: 190 px over 2 clicks" in text


def test_report_without_games():
    assert "no game yet" in game.report([], [])[0]
