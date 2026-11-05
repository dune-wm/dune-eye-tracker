import numpy as np

from dune_eye_tracker.live import OneEuro


def test_one_euro_steady_when_still_quick_when_moving():
    f = OneEuro()
    rng = np.random.default_rng(0)
    out = [f(np.array([1000.0, 500.0]) + rng.normal(0, 30, 2), i / 30) for i in range(60)]
    assert np.std([o[0] for o in out[30:]]) < 0.7 * 30  # the 30 px jitter calmed (as dune-eye-tracker's gaze filter)
    for i in range(60, 75):  # a jump of 800 px: followed within half a second
        o = f(np.array([1800.0, 500.0]), i / 30)
    assert o[0] > 1500


def test_shift_is_the_median_miss_on_y_only():
    from dune_eye_tracker.live import shift

    targets = np.array([[200.0, 100.0], [1280, 540], [2360, 980], [200, 980], [2360, 100]])
    guesses = targets + [-20, 130] + np.array([[5, -8], [0, 0], [-5, 8], [300, 400], [1, 2]])  # one point far off
    s = shift(guesses, targets)
    assert s[0] == 0.0  # x left as it is
    assert abs(s[1] - 130) < 10  # the outlier does not drag it
    both = shift(guesses, targets, axes=(True, True))
    assert abs(both[0] + 20) < 10
