import json

import numpy as np

from dune_eye_tracker import track
from dune_eye_tracker.measure import CALIBRATION
from test_model import fake_feats

SIZE = (2560, 1080)


def test_first_gaze_puts_the_pointer_there():
    p = track.Pointer(SIZE)
    assert p.update(0.0, (1000, 500), (0, 0)) == (1000, 500)


def test_gaze_near_keeps_it_and_the_head_moves_it():
    p = track.Pointer(SIZE, head_gain=2000)
    p.update(0.0, (1000, 500), (0.0, 0.0))
    # the gaze wobbles within the jump radius: nothing moves
    assert p.update(0.03, (1150, 420), (0.0, 0.0)) is None
    # the head turns right (head_x down) and down (head_y down): right and down
    assert p.update(0.06, (1100, 500), (-0.01, -0.005)) == (1020, 510)


def test_far_gaze_jumps_only_when_it_stays():
    p = track.Pointer(SIZE, jump=300, jump_time=0.15, head_gain=2000)
    p.update(0.0, (500, 500), (0, 0))
    assert p.update(0.03, (2000, 500), (0, 0)) is None  # one frame far: stays
    assert p.update(0.10, (2000, 500), (0, 0)) is None
    assert p.update(0.20, (2000, 520), (0, 0)) == (2000, 520)  # still far: jumps
    # and the head counts from the landing
    assert p.update(0.23, (2000, 520), (-0.01, 0)) == (2020, 520)


def test_far_then_back_near_does_not_jump():
    p = track.Pointer(SIZE, jump=300, jump_time=0.15)
    p.update(0.0, (500, 500), (0, 0))
    p.update(0.03, (2000, 500), (0, 0))
    p.update(0.06, (520, 500), (0, 0))  # back near: the clock restarts
    assert p.update(0.20, (2000, 500), (0, 0)) is None


def test_the_mouse_takes_the_pointer_until_the_gaze_moves_away():
    p = track.Pointer(SIZE, jump=300, jump_time=0.15, hand_time=0.5, head_gain=2000)
    p.update(0.0, (500, 500), (0, 0))
    # the hand moves it to where you look; the gaze guess is 400 px off from there
    p.hand(1.0, (900, 500))
    assert p.update(1.1, (1300, 500), (0, 0)) is None
    assert p.sent == (900, 500)
    # the hand stopped: the far guess, the head turning, do not take it back
    for t in (1.6, 1.8, 2.0, 2.5):
        assert p.update(t, (1300, 520), (-0.01, 0)) is None
    # looking elsewhere (the gaze moves past the jump) does, once it stays
    assert p.update(2.6, (2200, 500), (0, 0)) is None
    assert p.update(2.8, (2200, 500), (0, 0)) == (2200, 500)
    # and then the head moves it again
    assert p.update(2.83, (2200, 500), (-0.01, 0)) == (2220, 500)


def test_the_mouse_on_another_monitor_leaves_it_there():
    p = track.Pointer(SIZE, jump=300, jump_time=0.15, hand_time=0.5)
    p.update(0.0, (500, 500), (0, 0))
    p.hand(1.0, None)
    assert p.update(1.2, (500, 500), (0, 0)) is None
    assert p.update(2.0, (520, 500), (0, 0)) is None  # same gaze: not pulled back
    # looking elsewhere on this monitor, for long, does not take it either
    for t in (2.1, 2.3, 3.0, 5.0):
        assert p.update(t, (1500, 500), (0, 0)) is None
    # nor looking off the screen and back
    p.update(5.1, (1000, 1500), (0, -0.05))
    p.update(5.6, (1000, 1500), (0, -0.05))
    assert p.update(5.8, (1200, 700), (0, 0)) is None
    # the mouse back on this monitor: the hand's, then the gaze takes it as usual
    p.hand(6.0, (100, 100))
    assert p.update(6.1, (1200, 700), (0, 0)) is None
    assert p.sent == (100, 100)
    assert p.update(6.6, (1200, 700), (0, 0)) is None  # kept where the hand left it
    p.update(6.7, (2200, 500), (0, 0))
    assert p.update(6.9, (2200, 500), (0, 0)) == (2200, 500)


def test_kept_on_the_screen():
    p = track.Pointer(SIZE, head_gain=2000)
    p.update(0.0, (2550, 10), (0, 0))
    assert p.update(0.03, (2550, 10), (-0.01, 0.01)) == (2559, 0)


def test_calibration_fits_the_eyes(tmp_path):
    rng = np.random.default_rng(2)
    samples = [
        {"point": i, "target": [x, y], "feats": fake_feats(x, y, rng, 0.002)}
        for i, (x, y) in enumerate(CALIBRATION)
        for _ in range(20)
    ]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"size": list(SIZE), "output": "HDMI-A-2", "calibration": samples}))
    data, eyes = track.load_calibration(path)
    assert data["output"] == "HDMI-A-2"
    guess = eyes.predict([fake_feats(0.3, 0.6, rng)], SIZE)[0]
    assert np.linalg.norm(guess - (0.3 * SIZE[0], 0.6 * SIZE[1])) < 60


def test_move_speaks_dunes_protocol(tmp_path, monkeypatch):
    import socket

    path = str(tmp_path / "dune.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    monkeypatch.setenv("DUNE_SOCKET", path)
    track.move(12, 34, "HDMI-A-1")
    conn, _ = server.accept()
    assert conn.recv(100) == b"pointer\tmove\t12\t34\tHDMI-A-1"
    conn.close()
    server.close()


def test_looking_below_lets_go_and_back_takes_it():
    p = track.Pointer(SIZE, away=150, away_time=0.3)
    assert p.update(0.0, (1000, 500), (0, 0)) == (1000, 500)
    # the monitor below: a glance does nothing, staying there lets go
    assert p.update(0.1, (1000, 1500), (0, -0.05)) is None
    assert not p.released
    assert p.update(0.5, (1000, 1500), (0, -0.05)) is None
    assert p.released
    # nothing sent while away, even with the head turning
    assert p.update(0.6, (1300, 1600), (0.02, -0.06)) is None
    # near the edge is not back yet
    assert p.update(0.7, (1000, 1150), (0, -0.03)) is None
    assert p.released
    # well on the screen: the pointer jumps to the gaze
    assert p.update(0.8, (1200, 700), (0, 0)) == (1200, 700)
    assert not p.released


def test_a_glance_off_the_screen_keeps_the_pointer():
    p = track.Pointer(SIZE, away=150, away_time=0.3)
    p.update(0.0, (1000, 500), (0, 0))
    p.update(0.1, (1000, 1500), (0, 0))
    assert p.update(0.2, (1000, 520), (0, 0)) is None  # back near: no jump, no release
    assert not p.released


def test_head_turned_down_to_the_monitor_below_lets_go():
    # the eyes back in the middle, the head down (head_y below where it was): away
    p = track.Pointer(SIZE, away=150, away_time=0.3, head_ref=(0.0, -0.06), head_screen=4000)
    p.update(0.0, (1280, 540), (0.0, -0.06))
    p.update(0.1, (1280, 600), (0.0, -0.25))
    p.update(0.5, (1280, 600), (0.0, -0.25))
    assert p.released
    assert p.update(0.6, (1280, 540), (0.0, -0.06)) == (1280, 540)


def test_the_pointer_glides_instead_of_jumping():
    p = track.Pointer(SIZE, jump=300, jump_time=0.15, glide=0.1)
    assert p.update(0.0, (500, 500), (0, 0)) == (500, 500)  # the first: put there
    p.update(0.03, (2000, 500), (0, 0))
    p.update(0.10, (2000, 500), (0, 0))
    first = p.update(0.20, (2000, 500), (0, 0))  # the jump: only part of the way
    assert 500 < first[0] < 2000
    xs = [p.update(0.20 + i * 0.037, (2000, 500), (0, 0)) for i in range(1, 20)]
    xs = [x for x, _ in filter(None, xs)]
    assert xs == sorted(xs) and xs[-1] >= 1995  # on its way, without overshooting, then there
    # head jitter of a few px is smoothed, not sent as is
    p.update(0.93, (2000, 500), (0.002, 0))  # a frame of the head 4 px off
    assert p.sent[0] >= 1998


def test_a_head_turned_past_the_screen_steps_one_tile_until_it_comes_back():
    p = track.Pointer(SIZE, head_gain=2000, head_screen=0.0, step_at=(0.4, -0.3))
    p.update(0.0, (1280, 500), (0.0, 0.0))
    steps = []
    # the head turns left (head_x up) past 0.4 and stays there, the eyes in the middle
    for i in range(1, 35):
        p.update(0.03 * i, (1280, 500), (min(0.05 * i, 0.5), 0.0))
        steps.append(p.step)
    assert steps.count(-1) == 1 and steps.count(1) == 0  # one tile left, held: no more
    sent = p.sent
    # a little back is not back: still no step, the pointer stays
    p.update(1.1, (1280, 500), (0.3, 0.0))
    assert p.step == 0 and p.sent == sent
    # back towards the middle, then turned again: the next tile
    p.update(1.2, (900, 500), (0.1, 0.0))
    assert p.step == 0 and p.sent is not None and abs(p.sent[0] - 900) < 2
    p.update(1.3, (900, 500), (0.45, 0.0))
    assert p.step == -1
    # the right side steps right
    p.update(1.4, (900, 500), (0.0, 0.0))
    p.update(1.5, (900, 500), (-0.35, 0.0))
    assert p.step == 1


def test_looking_at_the_screen_with_the_head_turned_does_not_step():
    p = track.Pointer(SIZE, head_gain=2000, head_screen=0.0, step_at=(0.4, -0.3))
    for i, x in enumerate((0.0, 0.2, 0.35, -0.25, 0.0)):
        p.update(0.03 * i, (100, 500), (x, 0.0))
        assert p.step == 0


def test_step_turns_come_from_the_clicks():
    clicks = [{"feats": {"head_x": x}} for x in np.linspace(-0.25, 0.3, 300)]
    left, right = track.step_turns({"click_samples": clicks})
    assert 0.37 < left < 0.39 and -0.34 < right < -0.32
    assert track.step_turns({"click_samples": clicks[:50]}) is None
    p = track.Pointer(SIZE, head_ref=(0.02, 0.0))  # without: the calibration's head ± STEP_TURN
    assert p.step_at == (0.02 + track.STEP_TURN, 0.02 - track.STEP_TURN)


def test_action_speaks_dunes_protocol(tmp_path, monkeypatch):
    import socket

    path = str(tmp_path / "dune.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    monkeypatch.setenv("DUNE_SOCKET", path)
    track.action("focus_left")
    conn, _ = server.accept()
    assert conn.recv(100) == b"action\tfocus_left"
    conn.close()
    server.close()
