import math

import numpy as np

from dune_eye_tracker import record


def test_plan_every_block_in_turn_points_apart():
    p = record.plan((2560, 1080), np.random.default_rng(0))
    assert [b for b, *_ in p] == ["calibration"] * 28 + ["head"] * 5 + ["grid"] * 24
    grid = [t for b, _, t, _ in p if b == "grid"]
    far = [math.dist(a, b) >= record.APART for a, b in zip(grid, grid[1:])]
    assert sum(far) >= len(far) - 2
    assert not set(record.GRID) & set(record.CALIBRATION)
