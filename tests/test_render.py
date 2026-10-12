import numpy as np

from tcc.utils.render import BACKGROUND, BONE, draw_skeletons
from tests.fixtures.synthetic import generate


def _pose() -> np.ndarray:
    _, seq = generate(seed=2, n_train=1, n_test=0)[0].people[1]
    return seq[:1]


def test_canvas_size_follows_scale() -> None:
    img = draw_skeletons(_pose(), np.array([1]), (1920, 1080), scale=0.5, conf_min=0.1)
    assert img.shape == (540, 960, 3) and img.dtype == np.uint8


def test_empty_frame_is_plain_background() -> None:
    img = draw_skeletons(
        np.zeros((0, 17, 3), np.float32), np.zeros(0, np.int64), (100, 50), 1.0, 0.1
    )
    assert (img == np.array(BACKGROUND, np.uint8)).all()


def test_skeleton_draws_bones() -> None:
    img = draw_skeletons(_pose(), np.array([1]), (1920, 1080), scale=0.5, conf_min=0.1)
    assert (img == np.array(BONE, np.uint8)).all(-1).sum() > 50


def test_low_confidence_joints_are_not_drawn() -> None:
    pose = _pose().copy()
    pose[..., 2] = 0.0
    img = draw_skeletons(pose, np.array([1]), (1920, 1080), scale=0.5, conf_min=0.1)
    assert (img == np.array(BACKGROUND, np.uint8)).all()
