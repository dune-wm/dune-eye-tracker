import numpy as np

from conftest import BACK, IMAGE, KAPPA, frame, rot, to_plane
from dune_eye_tracker import face


def test_canonical_face_has_its_eyes_where_they_should_be(canon):
    assert canon.shape == (468, 3)
    r, l = face.eyeball(canon, face.RIGHT_EYE, 1.0), face.eyeball(canon, face.LEFT_EYE, 1.0)
    assert np.isclose(r[0], -l[0]) and r[0] < 0  # mirrored, the right eye at -x
    assert 5 < l[0] - r[0] < 8  # cm between the eyes
    assert canon[1][2] > r[2]  # the nose before the eyes


def test_a_ray_goes_back_through_the_point_it_came_from():
    pts = np.array([[3.0, -2.0, -40.0], [-10.0, 5.0, -55.0]])
    uv = face.project(pts, 16 / 9)
    d = face.ray(uv, 16 / 9)
    assert np.allclose(d, pts / np.linalg.norm(pts, axis=1, keepdims=True))


def test_the_pose_puts_the_canonical_face_on_its_landmarks(canon):
    lm, m = frame(canon, (1000, 500), head=(0.2, -0.1))
    assert face.reprojection(lm[None], m[None], IMAGE, canon)[0] < 1e-6


def test_the_gaze_meets_the_plane_where_the_target_is_whatever_the_head(canon):
    for head, pos, target in (((0, 0), (0, 3, -40), (100, 60)), ((0.3, -0.2), (-6, 1, -35), (2400, 1000)), ((-0.25, 0.15), (5, 6, -48), (1280, 540))):
        lm, m = frame(canon, target, head, pos)
        xy = face.on_plane(lm[None], m[None], IMAGE[0] / IMAGE[1], canon, BACK, KAPPA)[0]
        assert np.allclose(xy, to_plane(target), atol=1e-3)


def test_the_eyes_in_the_head_do_not_move_with_the_head(canon):
    # the same eyes-in-head turn, two head turns: the angles stay
    a = []
    for head in ((0.0, 0.0), (0.3, 0.1)):
        r = rot(*head)
        lm, m = frame(canon, (1280, 300), head, (0, 3, -40))
        a.append(face.eye_angles(lm[None], m[None], IMAGE[0] / IMAGE[1], canon, BACK)[0])
    # the target moved in the head's frame as the head turned: the eyes turned by about that
    assert abs((a[0][0] - a[1][0]) - 0.3) < 0.08
