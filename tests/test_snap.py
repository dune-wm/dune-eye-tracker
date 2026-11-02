import json

from dune_eye_tracker import track
from dune_eye_tracker.snap import nearest, parse

SIZE = (2560, 1080)


def test_nearest_small_target_lands_in_its_middle():
    ts = [(100, 100, 21, 21), (300, 100, 21, 21)]
    assert nearest(ts, 140, 110, 90) == (110, 110)
    assert nearest(ts, 200, 400, 90) is None


def test_inside_two_the_smaller_wins():
    ts = [(0, 0, 100, 100), (40, 40, 11, 11)]
    assert nearest(ts, 45, 45, 90) == (45, 45)


def test_a_big_target_keeps_the_gaze_inside():
    assert nearest([(0, 0, 600, 30)], 300, 60, 90) == (300, 25)


def test_parse_watch_lines():
    line = json.dumps({"output": "M", "targets": [{"x": 1, "y": 2, "w": 3, "h": 4, "role": "button", "name": "Ok"}]})
    assert parse(line) == [(1, 2, 3, 4)]
    assert parse("nope") is None
    assert parse(json.dumps({"output": "M"})) is None


def test_a_jump_lands_on_the_button_near_the_gaze():
    targets = [(1960, 480, 80, 30)]
    p = track.Pointer(SIZE, jump=300, jump_time=0.15, head_gain=2000, snap=lambda g: nearest(targets, g[0], g[1], 90))
    p.update(0.0, (500, 500), (0, 0))
    p.update(0.03, (2060, 560), (0, 0))
    # lands on the button's middle, not on the gaze 60 px below it
    assert p.update(0.20, (2060, 560), (0, 0)) == (2000, 494)
    # and the head moves it from there
    assert p.update(0.23, (2060, 560), (-0.01, 0)) == (2020, 494)


def test_no_button_near_lands_on_the_gaze():
    p = track.Pointer(SIZE, snap=lambda g: nearest([(0, 0, 10, 10)], g[0], g[1], 90))
    assert p.update(0.0, (1000, 500), (0, 0)) == (1000, 500)


def test_put_lines_for_this_monitor_only():
    from dune_eye_tracker.snap import parse_put

    assert parse_put("put 1200 340 HDMI-A-1\n", "HDMI-A-1") == (1200, 340)
    assert parse_put("put 1200.5 340 HDMI-A-1", "HDMI-A-1") == (1200, 340)
    assert parse_put("put 1200 340 HDMI-A-2", "HDMI-A-1") is None
    assert parse_put("move 1 2 HDMI-A-1", "HDMI-A-1") is None
    assert parse_put("put x 2 HDMI-A-1", "HDMI-A-1") is None


def test_a_put_stays_until_the_gaze_moves_away():
    p = track.Pointer(SIZE, jump=300, jump_time=0.15, hand_time=0.5)
    p.update(0.0, (500, 500), (0, 0))
    p.hand(0.1, (1500, 300))  # dune-a11y put it on a button
    assert p.update(0.2, (520, 500), (0, 0)) is None
    # the hand time over: the gaze where it was then keeps it there
    assert p.update(0.7, (520, 500), (0, 0)) is None
    assert p.sent == (1500, 300)
