import pytest

from dune_eye_tracker import face
from dune_eye_tracker.synthetic import BACK, IMAGE, KAPPA, PX_CM, SCREEN, frame, rot, synthetic, to_plane, with_iris  # noqa: F401


@pytest.fixture(scope="session")
def canon():
    if not face.model_path().exists():
        pytest.skip("no face_landmarker.task")
    return face.canonical()
