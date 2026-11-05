import numpy as np
import pytest

from dune_eye_tracker import gaze
from dune_eye_tracker.camera import Frame


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    from dune_eye_tracker import face
    from dune_eye_tracker.synthetic import synthetic, with_iris

    if not face.model_path().exists():
        pytest.skip("no face_landmarker.task")
    canon = face.canonical()
    rec = with_iris(synthetic(canon, np.random.default_rng(0), noise=0.0004))  # ~0.5 px (normalised)
    path = tmp_path_factory.mktemp("rec") / "session.npz"
    rec.save(path)
    return rec, path


def samples(rec, idx):
    return [
        Frame(int(i), float(rec.t[i]), rec.landmarks[i], rec.matrix[i], rec.image,
              {"dark": rec.dark[i], "limbus": rec.limbus[i], "maps": rec.maps[i]}, {"open": 0.3, "head_x": float(rec.matrix[i, 0, 2])})
        for i in idx
    ]


def test_a_dune_gaze_session_imported_drives_the_gaze_where_you_look(session, tmp_path):
    rec, path = session
    store = tmp_path / "gaze.npz"
    assert gaze.import_session(str(path), "HDMI-A-1", store) == 0
    g = gaze.load(store)
    assert g.output == "HDMI-A-1" and tuple(g.screen) == (2560, 1080)
    assert "quadratic" in g.model.name  # 28 calibration points
    grid = np.flatnonzero(rec.block == 2)
    guess = g.predict(samples(rec, grid))
    miss = [np.linalg.norm(np.median(guess[rec.point[grid] == p], axis=0) - rec.target[grid][rec.point[grid] == p][0])
            for p in np.unique(rec.point[grid])]
    assert np.median(miss) < 30  # each point's median: the head free, the geometry gives the gaze back
    g.offset = np.array([0.0, 100.0])
    assert np.allclose(g.predict(samples(rec, grid[:1]))[0], guess[0] - [0, 100])


def test_import_says_what_is_missing(session, tmp_path, capsys):
    rec, path = session
    assert gaze.import_session(str(path), None, tmp_path / "g.npz") == 1
    assert "--output" in capsys.readouterr().err


def test_no_iris_no_guess():
    assert gaze.one([Frame(1, 0.0, None, None, (1920, 1080))], (2560, 1080)) is None


def learner_on(session, tmp_path):
    rec, path = session
    store = tmp_path / "gaze.npz"
    gaze.import_session(str(path), "HDMI-A-1", store)
    return rec, gaze.GazeLearner(gaze.load(store), store), store


def click_at(learner, rec, point, dy=0.0, by="mouse", t0=[100.0]):
    """The grid point `point` looked at, then clicked `dy` px off it (as if the gaze had
    moved that much the other way); its frames seen just before the click."""
    idx = np.flatnonzero((rec.block == 2) & (rec.point == point))[:8]
    t0[0] += 10.0
    for k, s in enumerate(samples(rec, idx)):
        s.t = t0[0] - 0.3 + 0.03 * k
        learner.seen(s)
    x, y = rec.target[idx][0]
    return learner.click(t0[0], int(x), int(y + dy), "HDMI-A-1", by)


def test_clicks_add_points_and_the_mouse_ones_give_the_shift(session, tmp_path):
    rec, learner, store = learner_on(session, tmp_path)
    assert click_at(learner, rec, 0, by="mouse").startswith("kept")
    assert learner.click(1.0, 10, 10, "DP-1") == "other monitor"
    assert learner.click(5000.0, 10, 10, "HDMI-A-1") == "no iris before it"
    for p in range(1, 4):
        click_at(learner, rec, p, dy=-130)
    assert not learner.g.offset.any()  # fewer than SHIFT_FROM mouse clicks: none yet
    for p in range(4, 9):
        click_at(learner, rec, p, dy=-130, by={4: "name", 5: "command"}.get(p, "mouse"))  # the voice's do not count
    assert {learner.g.rec.meta["by"][str(i)] for i in learner.mouse_clicks()} == {"mouse"}
    assert learner.g.offset[0] == 0.0
    assert abs(learner.g.offset[1] - 130) < 20  # guessed 130 px below the clicks: taken off
    learner.save()
    again = gaze.load(store)
    assert len(np.unique(again.rec.point[again.rec.block == gaze.CLICK])) == 9
    assert again.rec.meta["by"]["4"] == "name"


def test_a_refit_on_the_clicks_is_checked_before_it_is_used(session, tmp_path):
    rec, learner, _ = learner_on(session, tmp_path)
    for p in range(10):
        click_at(learner, rec, p)
    job = learner.due()
    assert job is not None and learner.due() is None  # one fit per REFIT clicks
    model, note = gaze.GazeLearner.fit(*job)
    assert "clicks: on the last" in note
    learner.apply((model, note))


def test_no_virtual_camera_no_sharing_said_once(tmp_path):
    from dune_eye_tracker.camera import Camera

    cam = object.__new__(Camera)  # no real webcam: only the sharing
    cam.share, cam.shared = str(tmp_path / "video10"), None
    assert cam.open_share() is None
    assert "v4l2loopback not loaded" in cam.shared
    cam.share = None
    assert cam.open_share() is None


def test_sharing_goes_on_and_off_as_asked_while_the_camera_runs():
    import threading
    import types

    import cv2
    import numpy as np

    from dune_eye_tracker.camera import Camera

    sent, closed = [], []

    class Out:
        def send(self, frame):
            sent.append(n[0])

        def close(self):
            closed.append(n[0])

    cam = object.__new__(Camera)  # a fake webcam: frames of 8x8 px, no face
    threading.Thread.__init__(cam, daemon=True)
    n = [0]
    # what is asked before each frame: off at first, on from frame 3, off from frame 6
    asks = {3: True, 6: False}

    def read():
        n[0] += 1
        if n[0] in asks:
            cam.set_sharing(asks[n[0]])
        return (n[0] <= 8), np.zeros((8, 8, 3), np.uint8)

    cam.cv2, cam.cap = cv2, types.SimpleNamespace(read=read, release=lambda: None)
    cam.share, cam.shared, cam.keep_crops = "/dev/video10", None, False
    cam.sharing, cam.stopped, cam.lock = threading.Event(), threading.Event(), threading.Lock()
    cam.error, cam.fps = None, 0.0
    cam.landmarker = lambda: types.SimpleNamespace(
        detect_for_video=lambda *a: types.SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[]), close=lambda: None)
    cam.open_share = lambda: Out()
    cam.run()
    assert sent == [3, 4, 5] and closed == [6]
    assert "any more" in cam.shared
    cam.share = None
    assert not cam.set_sharing(True)


def test_the_head_ref_and_the_tile_steps_come_from_the_calibration(session, tmp_path):
    rec, learner, _ = learner_on(session, tmp_path)
    ref = learner.g.head_ref()
    cal = rec.block == 0
    assert abs(ref[0] - np.median(rec.matrix[cal, 0, 2])) < 1e-6
    left, right = learner.g.step_turns()  # too few clicks: from the calibration's head
    assert abs(left - (ref[0] + gaze.STEP_TURN)) < 1e-6 and abs(right - (ref[0] - gaze.STEP_TURN)) < 1e-6


def test_each_click_is_a_line_of_the_click_log(session, tmp_path):
    import json

    rec, learner, store = learner_on(session, tmp_path)
    click_at(learner, rec, 3)
    line = json.loads((store.parent / "clicks.jsonl").read_text().splitlines()[-1])
    assert line["said"].startswith("kept") and line["seen"] and line["guess"] and "head_x" in line["feats"]
    jump = {"kind": "gaze", "landed": [800.0, 430.0], "age": 0.4}  # what the tracker knew: logged only
    assert learner.click(5000.0, 812, 440, "HDMI-A-1", "mouse", {"by": "mouse", "jump": jump}) == "no iris before it"
    line = json.loads((store.parent / "clicks.jsonl").read_text().splitlines()[-1])
    assert line["jump"] == jump and line["by"] == "mouse" and line["said"] == "no iris before it"


def test_the_tile_steps_come_from_the_clicks_once_there_are_enough():
    from dune_eye_tracker.recording import Recording

    n = 250
    m = np.tile(np.eye(4), (n, 1, 1))
    m[:, 0, 2] = np.linspace(-0.25, 0.3, n)  # head_x of each click
    rec = Recording(np.zeros((n, 478, 3)), m, np.arange(n, dtype=float), np.full(n, gaze.CLICK), np.arange(n), np.zeros((n, 2)),
                    np.full(n, -0.2), {"screen": [2560, 1080], "image": [1920, 1080]})
    left, right = gaze.Gaze(rec, model=object()).step_turns()
    assert abs(left - (np.percentile(m[:, 0, 2], 99) + gaze.STEP_MARGIN)) < 1e-9
    assert abs(right - (np.percentile(m[:, 0, 2], 1) - gaze.STEP_MARGIN)) < 1e-9
