import numpy as np
import pytest

from dune_eye_tracker import iris


def eye_image(cx, cy, r=19.0, lid_top=14.0, lid_bottom=10.0, noise=3.0, seed=0, shape=iris.CROP):
    """A crop: skin, an almond of sclera, the iris (dark, a darker pupil) at (cx, cy),
    the lids covering it above and below; drawn 4x finer and averaged (subpixel)."""
    w, h = shape
    ss = 4
    yy, xx = (np.mgrid[0 : h * ss, 0 : w * ss] + 0.5) / ss - 0.5  # pixel centres on the integers (as OpenCV)
    img = np.full(xx.shape, 150.0)  # skin
    ex, ey = w / 2, h / 2
    almond = ((xx - ex) / 48) ** 2 + ((yy - ey) / 22) ** 2 <= 1
    img[almond] = 215.0  # sclera
    d = np.hypot(xx - cx, yy - cy)
    open_ = (yy > cy - lid_top) & (yy < cy + lid_bottom)
    img[(d <= r) & almond & open_] = 70.0
    img[(d <= r * 0.4) & almond & open_] = 25.0
    img = img.reshape(h, ss, w, ss).mean(axis=(1, 3))
    return np.clip(img + np.random.default_rng(seed).normal(0, noise, img.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("cx, cy", [(80.0, 50.0), (71.3, 48.6), (92.7, 52.25), (64.4, 49.9)])
def test_limbus_finds_the_centre_to_a_fraction_of_a_pixel(cx, cy):
    img = eye_image(cx, cy)
    start = np.array([cx + 2.5, cy - 1.5])  # MediaPipe a little off
    lb = iris.limbus(img, start, 19.0)
    assert abs(lb[0] - cx) < 0.1  # across: both sides of the iris seen
    assert abs(lb[1] - cy) < 0.5  # up / down: the lids hide its top and bottom
    assert abs(lb[2] - 19.0) < 0.6


def test_limbus_with_a_wrong_start_and_radius_still_gets_there():
    img = eye_image(75.2, 50.4)
    lb = iris.limbus(img, np.array([79.0, 47.0]), 16.0)
    assert abs(lb[0] - 75.2) < 0.1 and abs(lb[1] - 50.4) < 0.5


def test_dark_centroid_keeps_to_the_pupil_the_lids_pull_it_a_little():
    # the lids cover more above: the visible dark part's centroid sits a little low, the
    # darkest quarter (the pupil) holding it near the centre
    img = eye_image(80.0, 50.0, lid_top=8.0, lid_bottom=14.0)
    dk = iris.dark(img, np.array([80.0, 50.0]), 19.0)
    assert abs(dk[0] - 80.0) < 0.3
    assert 50.0 < dk[1] < 51.0


def test_crop_map_round_trips_and_levels_the_eye():
    left, right = np.array([900.0, 500.0]), np.array([968.0, 510.0])
    a = iris.crop_affine(left, right)
    back = iris.apply(iris.invert(a), np.array([left, right]))
    assert np.allclose(back[:, 1], iris.CROP[1] / 2)  # level
    assert np.isclose(back[1, 0] - back[0, 0], iris.EYE_PX)  # 90 px wide
    p = np.array([[10.0, 20.0]])
    assert np.allclose(iris.apply(iris.invert(a), iris.apply(a, p)), p)


def test_a_recording_with_eye_crops_gives_the_iris_back_in_image_pixels(canon):
    from conftest import synthetic
    from dune_eye_tracker import face, models

    rec = synthetic(canon, np.random.default_rng(4))
    n = 6
    rec.landmarks, rec.matrix, rec.t = rec.landmarks[:n], rec.matrix[:n], rec.t[:n]
    w, h = rec.image
    crops, maps, truth = [], [], []
    for k in range(n):
        px = rec.landmarks[k, :, :2] * np.array([w, h])
        cs, ms, ts = [], [], []
        for eye in face.EYES:
            a = iris.crop_affine(px[eye["left"]], px[eye["right"]])
            c = np.array([80.0 + 0.37 * k, 50.0 - 0.21 * k])  # the iris in the crop
            start = c - np.array([2.0, -1.0])  # MediaPipe a little off
            rec.landmarks[k, eye["iris"], :2] = iris.apply(a, start[None])[0] / np.array([w, h])
            cs.append(eye_image(*c, seed=k))
            ms.append(a)
            ts.append(iris.apply(a, c[None])[0])
        crops.append(cs), maps.append(ms), truth.append(ts)
    rec.crops, rec.maps = np.array(crops), np.array(maps)
    rec.limbus, rec.dark = np.zeros((n, 2, 4)), np.zeros((n, 2, 2))
    lm = models.landmarks(rec, np.arange(n), "limbus")
    got = np.stack([lm[:, e["iris"], :2] * np.array([w, h]) for e in face.EYES], 1)
    scale = np.sqrt(abs(np.linalg.det(rec.maps[0, 0, :, :2])))  # image px per crop px
    assert np.max(np.abs(got - np.array(truth))) < 1.0 * scale  # the plumbing: crop to image and back
