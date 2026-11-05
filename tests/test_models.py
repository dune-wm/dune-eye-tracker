import numpy as np

from conftest import synthetic
from dune_eye_tracker import evaluate
from dune_eye_tracker.models import Geometric, Ridge, tracker
from dune_eye_tracker.recording import Recording


def fit_test(model, rec):
    idx = evaluate.frames(rec, "calibration")
    model.fit(rec, idx, evaluate.groups(rec, idx))
    return np.median([d for _, d in evaluate.misses(model, rec, "grid")])


def test_geometric_finds_the_eyes_and_the_screen_whatever_the_head(canon):
    rec = synthetic(canon, np.random.default_rng(1))
    g = Geometric("g", 1)
    assert fit_test(g, rec) < 5
    assert abs(g.p[0] - 1.1) < 0.2  # the eyeball's depth found back


def test_the_2d_eyes_model_is_thrown_by_the_head(canon):
    rec = synthetic(canon, np.random.default_rng(1))
    flat = Ridge("eyes", ("eye_u", "eye_v"), ("eye_u", "eye_v"), tracker)
    assert fit_test(flat, rec) > 5 * fit_test(Geometric("g", 1), rec)


def test_recording_saves_and_loads(canon, tmp_path):
    rec = synthetic(canon, np.random.default_rng(2))
    rec.save(tmp_path / "r.npz")
    back = Recording.load(tmp_path / "r.npz")
    assert back.screen == rec.screen and back.image == rec.image and len(back.t) == len(rec.t)
    assert np.allclose(back.landmarks, rec.landmarks, atol=1e-6)


def test_report_within_and_across(canon):
    rng = np.random.default_rng(3)
    a, b = synthetic(canon, rng, noise=0.0005), synthetic(canon, rng, noise=0.0005)
    b.name = "other"
    text = "\n".join(evaluate.report([a, b]))
    assert "28 calibration, 5 head-moving, 24 grid points" in text
    assert "across recordings" in text and "geometric " in text


def test_ci():
    lo, hi = evaluate.ci(np.random.default_rng(0).normal(150, 30, 40))
    assert lo < 150 < hi
    assert evaluate.ci([1, 2]) is None


def test_binocular_fits_each_eye_and_blends_them(canon):
    from dune_eye_tracker.models import Binocular

    rec = synthetic(canon, np.random.default_rng(6))
    b = Binocular("b", "mediapipe")
    assert fit_test(b, rec) < 8
    assert all(k in Binocular.KS for k in b.k)


def test_guesses_are_kept_on_the_screen():
    g = evaluate.on_screen(np.array([[-300.0, 50.0], [3000.0, 1500.0], [100.0, 200.0]]), (2560, 1080))
    assert np.allclose(g, [[0, 50], [2559, 1079], [100, 200]])
