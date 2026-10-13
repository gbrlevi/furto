import numpy as np

from tcc.utils.render import (
    BACKGROUND,
    BONE,
    GREEN,
    NO_SCORE,
    RED,
    annotate_frame,
    blank_canvas,
    draw_skeletons,
    score_color,
)
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


def test_score_color_goes_from_green_to_red() -> None:
    assert score_color(0.0, 1.0) == GREEN
    assert score_color(1.0, 1.0) == RED
    assert score_color(5.0, 1.0) == RED
    assert score_color(float("nan"), 1.0) == NO_SCORE
    mid = score_color(0.5, 1.0)
    assert GREEN[1] > mid[1] > RED[1] or GREEN[1] < mid[1] < RED[1]


def test_annotate_frame_draws_box_and_alert_banner() -> None:
    pose = _pose()
    canvas = blank_canvas((1920, 1080), 0.5)
    box = np.array(
        [[pose[0, :, 0].min(), pose[0, :, 1].min(), pose[0, :, 0].max(), pose[0, :, 1].max()]],
        np.float32,
    )
    img = annotate_frame(
        canvas, pose, np.array([1]), box, np.array([2.0]), 1.0, 0.5, 0.1, alert=True
    )
    is_red = (img == np.array(RED, np.uint8)).all(-1)
    assert is_red[:28].mean() > 0.5  # banner
    assert is_red[28:].sum() > 50  # red box: score above threshold
    assert (canvas == np.array(BACKGROUND, np.uint8)).all()  # canvas untouched


def test_annotate_frame_without_alert_has_no_banner() -> None:
    canvas = blank_canvas((320, 240), 1.0)
    empty = np.zeros((0, 17, 3), np.float32)
    img = annotate_frame(canvas, empty, np.zeros(0, np.int64), np.zeros((0, 4), np.float32),
                         np.zeros(0), 1.0, 1.0, 0.1, alert=False)  # fmt: skip
    assert (img == np.array(BACKGROUND, np.uint8)).all()
