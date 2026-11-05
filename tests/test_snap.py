from dune_eye_tracker import track
from dune_eye_tracker.snap import parse_answer

SIZE = (2560, 1080)


def test_answers():
    assert parse_answer("7 1200 340\n") == (7, (1200.0, 340.0), None)
    assert parse_answer("7 -") == (7, None, None)
    around = [{"x": 1, "y": 2, "w": 30, "h": 20, "role": "link"}]
    assert parse_answer('7 1200 340 {"around":[{"x":1,"y":2,"w":30,"h":20,"role":"link"}]}\n') == (7, (1200.0, 340.0), around)
    assert parse_answer('7 - {"around":[]}') == (7, None, [])
    assert parse_answer("7 - {broken") is None
    assert parse_answer("x 1 2") is None
    assert parse_answer("") is None


def test_a_jump_glides_onto_the_button_when_the_answer_comes():
    p = track.Pointer(SIZE)
    p.update(0.0, (500, 500), (0, 0))
    for i in range(1, 14):  # the gaze goes there and holds: landing 2
        p.update(0.03 * i, (2060, 560), (0, 0))
    assert p.landings == 2 and p.sent == (2060, 560)
    assert p.snap_to(2, (2000, 494))  # dune-a11y: the button is there
    assert p.update(0.42, (2060, 560), (0, 0)) == (2000, 494)
    # the gaze's wobble does not move it from there, the head does
    assert p.update(0.45, (2090, 580), (0, 0)) is None
    assert p.update(0.48, (2090, 580), (-0.01, 0)) == (2012, 494)


def test_a_late_answer_for_an_older_jump_is_dropped():
    p = track.Pointer(SIZE)
    p.update(0.0, (500, 500), (0, 0))  # jump 1
    for i in range(1, 14):
        p.update(0.03 * i, (2060, 560), (0, 0))  # jump 2
    assert not p.snap_to(1, (520, 500))
    assert p.landed.tolist() == [2060, 560]


def test_no_snap_while_the_hand_has_it():
    p = track.Pointer(SIZE, hand_time=0.5)
    p.update(0.0, (1000, 500), (0, 0))
    p.hand(0.1, (1200, 400))
    assert not p.snap_to(p.landings, (1010, 505))


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
