import json

from dune_eye_tracker import trace


def frames(tr, t0, n, at, raw=(1000, 500)):
    for k in range(n):
        tr.frame(t0 + k / 30, raw, raw, (0.0, -0.1), at, False, 0.3)


def test_there_and_back_is_a_bounce_with_its_frames(tmp_path):
    tr = trace.Trace(tmp_path / trace.LOG)
    frames(tr, 0.0, 10, (500, 500))
    assert not tr.jump(0.4, "gaze", (500, 500), (1300, 500))  # 800 px away
    frames(tr, 0.4, 10, (1300, 500), raw=(1300, 520))
    assert tr.jump(1.0, "gaze", (1300, 500), (560, 480))  # back near where it was, 0.6 s later
    line = json.loads((tmp_path / trace.LOG).read_text().splitlines()[-1])
    assert line["kind"] == "bounce" and line["went"] == [1300.0, 500.0] and line["back_after"] == 0.6
    assert len(line["frames"]) == 20 and line["frames"][-1]["at"] == [1300.0, 500.0]
    assert "bounce" in trace.summary(tr.lines)[0] and "1 bounces" in trace.summary(tr.lines)[0]


def test_going_on_or_coming_back_late_is_no_bounce():
    tr = trace.Trace(None)
    tr.jump(0.0, "gaze", (500, 500), (1300, 500))
    assert not tr.jump(0.5, "gaze", (1300, 500), (2000, 900))  # on elsewhere
    assert not tr.jump(2.5, "gaze", (2000, 900), (1300, 520))  # back where it was, but 2 s later
    short = trace.Trace(None)
    short.jump(0.0, "gaze", (2000, 880), (2100, 880))  # a short one (100 px): the head's last pixels
    assert not short.jump(0.2, "gaze", (2100, 880), (2010, 880))
    assert tr.bounces == short.bounces == 0


def test_the_pointers_speed_between_frames():
    tr = trace.Trace(None)
    tr.frame(0.0, (0, 0), (0, 0), (0, 0), (100, 100), False, 0.3)
    tr.frame(0.1, (0, 0), (0, 0), (0, 0), (400, 500), False, 0.3)
    assert tr.frames[-1]["speed"] == 5000  # 500 px in 0.1 s
    tr.frame(0.2, None, None, (0, 0), None, True, 0.05)  # let go, blinking: no speed
    assert tr.frames[-1]["speed"] is None and tr.frames[-1]["raw"] is None


def test_events_are_written_and_a_bad_path_does_not_stop_it(tmp_path):
    tr = trace.Trace(tmp_path / "missing" / trace.LOG)
    tr.event(1.0, "shift", note="shift +40 px")
    assert tr.lines[-1]["what"] == "shift"
    assert trace.summary([])[0].startswith("no jumps")
