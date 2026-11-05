from dune_eye_tracker import track

SIZE = (2560, 1080)


def frames(p, t0, gaze, n=8, head=(0, 0)):
    """`n` frames of the same gaze from `t0`, 0.03 s apart: the last sent."""
    out = None
    for i in range(n):
        out = p.update(t0 + 0.03 * i, gaze, head) or out
    return out


def test_first_gaze_puts_the_pointer_there():
    p = track.Pointer(SIZE)
    assert p.update(0.0, (1000, 500), (0, 0)) == (1000, 500)


def test_gaze_wobbling_keeps_it_and_the_head_moves_it():
    p = track.Pointer(SIZE, head_gain=2000)
    p.update(0.0, (1000, 500), (0.0, 0.0))
    # the gaze wobbles within `DEAD` of where it landed: nothing moves
    assert p.update(0.03, (1100, 440), (0.0, 0.0)) is None
    assert p.update(0.06, (1050, 560), (0.0, 0.0)) is None
    # the head turns right (head_x down) and down (head_y down): right and down
    assert p.update(0.09, (1000, 500), (-0.01, -0.005)) == (1020, 510)


def test_the_head_counts_from_each_landing():
    p = track.Pointer(SIZE, head_gain=2000)
    p.update(0.0, (500, 500), (0.0, 0.0))
    p.update(0.03, (500, 500), (-0.01, 0.0))  # 20 px right
    assert frames(p, 0.06, (1700, 520), n=12, head=(-0.01, 0.0)) == (1700, 520)  # landed: on the gaze
    assert p.update(0.5, (1700, 520), (-0.02, 0.0)) == (1720, 520)  # the head from there


def test_the_gaze_held_elsewhere_moves_it_there_at_once():
    p = track.Pointer(SIZE, dead=120, settle=40, settle_time=0.3)
    p.update(0.0, (500, 500), (0, 0))
    sent = [p.update(0.03 * i, (1700, 520), (0, 0)) for i in range(1, 15)]
    # nothing until the eyes have held still there 0.3 s, then one move, there
    assert sent[:9] == [None] * 9 and [s for s in sent if s] == [(1700, 520)]
    assert p.landings == 2 and p.landed.tolist() == [1700, 520] and p.came_from.tolist() == [500, 500]


def test_reading_or_scanning_does_not_move_it():
    p = track.Pointer(SIZE)
    p.update(0.0, (500, 500), (0, 0))
    # the eyes sweep across a line, never still: the pointer stays
    for i in range(1, 30):
        assert p.update(0.03 * i, (500 + 50 * i, 500), (0, 0)) is None
    assert p.landings == 1


def test_a_fixation_near_where_it_landed_does_not_move_it():
    p = track.Pointer(SIZE, dead=120)
    p.update(0.0, (500, 500), (0, 0))
    assert frames(p, 0.03, (600, 540), n=30) is None
    assert p.landings == 1


def test_the_mouse_takes_the_pointer_until_the_gaze_moves_away():
    p = track.Pointer(SIZE, jump=300, jump_time=0.15, hand_time=0.5)
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
    assert p.update(2.8, (2200, 500), (0, 0)) is None
    # held there: the gaze has it again
    assert frames(p, 2.83, (2200, 500), n=12) == (2200, 500)
    assert p.land_kind == "gaze"


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
    assert frames(p, 6.7, (2200, 500), n=20) == (2200, 500)


def test_kept_on_the_screen():
    p = track.Pointer(SIZE)
    assert p.update(0.0, (2600, -20), (0, 0)) == (2559, 0)


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


def test_the_pointer_glides_instead_of_jumping():
    p = track.Pointer(SIZE, glide=0.1)
    assert p.update(0.0, (500, 500), (0, 0)) == (500, 500)  # the first: put there
    sent = [p.update(0.03 * i, (2000, 500), (0, 0)) for i in range(1, 13)]
    first = next(x for x in sent if x)  # held there: off, only part of the way
    assert 500 < first[0] < 2000
    xs = [p.update(0.36 + i * 0.037, (2000, 500), (0, 0)) for i in range(1, 20)]
    xs = [x for x, _ in filter(None, xs)]
    assert xs == sorted(xs) and xs[-1] >= 1995  # on its way, without overshooting, then there
    # the gaze's wobble once there does not move it
    p.update(1.1, (2040, 530), (0, 0))
    assert p.at.tolist() == [2000, 500] and abs(p.sent[0] - 2000) <= 1


def test_a_head_turned_past_the_screen_steps_one_tile_until_it_comes_back():
    p = track.Pointer(SIZE, step_at=(0.4, -0.3), step_hold=0.0)
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
    p = track.Pointer(SIZE, step_at=(0.4, -0.3))
    for i, x in enumerate((0.0, 0.2, 0.35, -0.25, 0.0)):
        p.update(0.03 * i, (100, 500), (x, 0.0))
        assert p.step == 0


def test_without_step_turns_the_calibrations_head_and_step_turn():
    p = track.Pointer(SIZE, head_ref=(0.02, 0.0))
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


def test_the_gaze_on_the_zoom_is_mapped_back_to_real_pixels():
    zoom = {"x": 800, "y": 400, "scale": 3, "area_x": 0, "area_y": 0, "area_w": 1600, "area_h": 900}
    assert track.unzoom((860, 370), zoom).tolist() == [820, 390]
    assert track.unzoom((800, 400), zoom).tolist() == [800, 400]
    assert track.unzoom((2000, 370), zoom).tolist() == [2000, 370]  # outside the zoomed tile: as it is
    assert list(track.unzoom((860, 370), None)) == [860, 370]


def test_a_click_says_how_the_pointer_got_there():
    p = track.Pointer(SIZE, head_gain=2000)
    assert p.landing(0.0) is None  # nowhere yet
    p.update(0.0, (-60, 500), (0, 0))  # past the left edge: landed on it, the gaze kept as it was
    p.update(0.5, (-60, 500), (-0.01, 0))  # then the head moves it 20 px right
    j = p.landing(0.6)
    assert j["kind"] == "gaze" and j["landed"] == [0.0, 500.0] and j["gaze"] == [-60.0, 500.0]
    assert j["head"] == [20.0, 0.0] and j["at"] == [20.0, 500.0] and j["age"] == 0.6
    assert "snap" not in j
    frames(p, 0.7, (2060, 560), n=14, head=(-0.01, 0))  # landing 2, and dune-a11y moves it onto a button
    assert p.landings == 2
    assert p.snap_to(2, (2000, 494))
    j = p.landing(1.2)
    assert j["landed"] == [2000.0, 494.0] and j["gaze"] == [2060.0, 560.0] and j["snap"] == [-60.0, -66.0]


def test_a_click_after_the_hand_or_a_put_says_so():
    p = track.Pointer(SIZE, hand_time=0.5)
    p.update(0.0, (500, 500), (0, 0))
    p.hand(0.1, (900, 300))
    p.update(0.2, (500, 500), (0, 0))
    j = p.landing(0.3)
    assert j["kind"] == "hand" and j["landed"] == [900.0, 300.0] and "gaze" not in j
    p.hand(1.0, (100, 100), "put")
    p.update(1.1, (500, 500), (0, 0))
    assert p.landing(1.2)["kind"] == "put"


def test_back_from_a_head_step_lands_as_back():
    p = track.Pointer(SIZE, step_at=(0.4, -0.3), step_hold=0.0)
    p.update(0.0, (1000, 500), (0, 0))
    p.update(0.1, (1000, 500), (0.5, 0))  # turned: one tile left
    p.update(0.2, (1200, 500), (0.0, 0))  # back: lands on the gaze
    p.update(0.25, (1200, 500), (0.0, 0))
    assert p.landing(0.3)["kind"] == "back"


def test_after_the_hand_the_jump_it_put_right_is_kept():
    p = track.Pointer(SIZE, hand_time=0.5)
    p.update(0.0, (500, 500), (0, 0))
    frames(p, 0.03, (2060, 560), n=14)  # landing 2
    assert p.landings == 2
    assert p.snap_to(2, (2000, 494))
    p.hand(0.50, (1700, 640))  # then the mouse puts it right
    p.hand(0.55, (1690, 650))
    p.update(0.60, (2060, 560), (0, 0))
    j = p.landing(0.70)
    assert j["kind"] == "hand" and j["landed"] == [1690.0, 650.0]
    b = j["before"]
    assert b["kind"] == "gaze" and b["landed"] == [2000.0, 494.0] and b["gaze"] == [2060.0, 560.0]
    assert b["snap"] == [-60.0, -66.0] and b["at"] == [2000.0, 494.0] and b["head"] == [0.0, 0.0]
    assert p.before_seq == 2
    p.hand(1.1, (1600, 600))  # moved again: still that jump
    p.update(1.2, (1600, 600), (0, 0))
    assert p.landing(1.3)["before"]["landed"] == [2000.0, 494.0]


def test_a_new_jump_forgets_the_one_before_the_hand():
    p = track.Pointer(SIZE, jump=300, jump_time=0.15, hand_time=0.5)
    p.update(0.0, (500, 500), (0, 0))
    p.hand(0.1, (900, 300))
    p.update(0.2, (500, 500), (0, 0))
    assert p.landing(0.3)["before"]["landed"] == [500.0, 500.0]
    p.update(0.8, (500, 500), (0, 0))  # the hand gone, the gaze where it was then
    frames(p, 0.9, (1800, 300), n=10)  # then it moves away and settles: a landing
    j = p.landing(1.3)
    assert j["kind"] == "gaze" and "before" not in j and p.before is None


def test_a_turn_steps_only_once_held():
    p = track.Pointer(SIZE, step_at=(0.4, -0.3), step_hold=0.25)
    p.update(0.0, (1280, 500), (0.0, 0.0))
    p.update(0.1, (1280, 500), (0.45, 0.0))  # past it: not yet
    p.update(0.2, (1280, 500), (0.45, 0.0))
    assert p.step == 0 and p.turned == 0
    p.update(0.3, (1280, 500), (0.3, 0.0))  # back under it before the time: nothing
    p.update(0.4, (1280, 500), (0.45, 0.0))  # past again: the time starts again
    p.update(0.6, (1280, 500), (0.45, 0.0))
    assert p.step == 0
    p.update(0.66, (1280, 500), (0.45, 0.0))  # held 0.26 s: one tile left
    assert p.step == -1


def test_the_gaze_keeps_the_pointer_off_dunes_outer_gap_and_bars():
    one = {"tiles": [{"tile": 13, "x": 7, "y": 7, "w": 1270, "h": 1066}, {"tile": 37, "x": 1284, "y": 7, "w": 1270, "h": 1066}]}
    assert list(track.usable(one, SIZE)) == [7, 7, 2553, 1072]  # gap 7 all round (6 on the right)
    bar = {"tiles": [{"tile": 5, "x": 7, "y": 31, "w": 1906, "h": 1042}]}
    assert list(track.usable(bar, (1920, 1080))) == [7, 31, 1912, 1072]  # a bar's band on top
    scrolled = {"tiles": [{"tile": 1, "x": -600, "y": 7, "w": 1270, "h": 1066}, {"tile": 2, "x": 677, "y": 7, "w": 1270, "h": 1066},
                          {"tile": 3, "x": 1954, "y": 7, "w": 1270, "h": 1066}, {"tile": 4, "x": 4000, "y": 7, "w": 900, "h": 1066}]}
    assert list(track.usable(scrolled, SIZE)) == [0, 7, 2559, 1072]  # tiles cut by both edges: nothing kept off there
    along = {"view": 1282, "tiles": [{"tile": 3, "x": 8, "y": 8, "w": 1268, "h": 1064}, {"tile": 5, "x": 1290, "y": 8, "w": 1268, "h": 1064},
                                      {"tile": 6, "x": 2572, "y": 8, "w": 1268, "h": 1064}]}
    assert list(track.usable(along, SIZE)) == [8, 8, 2557, 1071]  # x along the strip, less the view
    lone = {"tiles": [{"tile": 1, "x": 7, "y": 7, "w": 1270, "h": 1066}]}
    assert track.usable(lone, SIZE)[2] == 2559 - track.EDGE_MAX  # the empty half is not a gap
    assert list(track.usable(None, SIZE)) == [0, 0, 2559, 1079]


def test_jumps_and_the_head_stay_in_the_area_the_hand_goes_anywhere():
    p = track.Pointer(SIZE)
    p.area = track.usable({"tiles": [{"tile": 1, "x": 7, "y": 7, "w": 2546, "h": 1066}]}, SIZE)
    assert p.update(0.0, (2, 1078), (0, 0)) == (7, 1072)  # looked into the gap: on the tile's edge
    frames(p, 0.1, (-300, 1300))  # the gaze further out (not long enough to let go): still in
    assert p.at[0] >= 7 and p.at[1] <= 1072
    p.hand(0.2, (2, 2))
    assert p.update(0.25, (10, 1070), (0, 0)) is None and list(p.hand_at) == [2, 2]
