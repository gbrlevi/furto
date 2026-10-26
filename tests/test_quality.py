"""Quality gate (ADR-0014): measures per window and the reason a window is not scorable."""

from pathlib import Path

import numpy as np
import pytest

from tcc.data.poselift import load_video
from tcc.data.skeleton import HIPS, SHOULDERS
from tcc.data.split import official_split
from tcc.preprocess.quality import (
    CRITERIA,
    REASONS,
    SCORABLE,
    failures,
    gate_reasons,
    torso_box_ratio,
    touches_border,
    window_quality,
)
from tcc.preprocess.trajectories import Trajectory, split_trajectories
from tcc.preprocess.windows import switch, treated_windows, video_windows
from tests.fixtures.synthetic import generate, write_poselift

CFG = {"keypoint_order": "yx_stored", "bbox_format": "xyxy", "nan_confidence": "interpolate"}

SIZE = (1920, 1080)
THRESHOLDS = {
    "mean_confidence": 0.5,
    "real_fraction": 0.8,
    "torso_fraction": 0.8,
    "box_height": 100.0,
    "border_fraction": 0.2,
}


def _clean_window() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A (1, 24) window of a synthetic walker, inside a 1920 x 1080 image."""
    _, seq = generate(seed=0, n_train=1, n_test=0)[0].people[1]
    kp = seq[:24].copy()
    kp[..., 2] = 0.9
    lo, hi = kp[..., :2].min(1), kp[..., :2].max(1)
    boxes = np.concatenate([lo - 5, hi + 5], axis=-1).astype(np.float32)
    return kp[None], boxes[None], np.ones((1, 24), np.bool_)


def _quality(kp: np.ndarray, boxes: np.ndarray, real: np.ndarray) -> dict[str, np.ndarray]:
    return window_quality(kp, boxes, real, SIZE, conf_min=0.1, border_margin=2.0)


def test_clean_window_passes_the_gate() -> None:
    kp, boxes, real = _clean_window()
    q = _quality(kp, boxes, real)
    assert q["box_height"][0] > THRESHOLDS["box_height"]
    assert gate_reasons(q, THRESHOLDS).tolist() == [SCORABLE]


def _break(criterion: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kp, boxes, real = _clean_window()
    if criterion == "mean_confidence":
        kp[..., [0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16], 2] = 0.05  # torso stays valid
    elif criterion == "real_fraction":
        real[:, ::2] = False
    elif criterion == "torso_fraction":
        kp[:, :, SHOULDERS, 2] = 0.05
        kp[:, :, [0, 1, 2, 3, 4], 2] = 1.0  # keep the mean confidence above 0.5
    elif criterion == "box_height":
        mid = (boxes[..., 1] + boxes[..., 3]) / 2
        boxes[..., 1], boxes[..., 3] = mid - 40, mid + 40
    elif criterion == "border_fraction":
        boxes[:, :12, 0] = 1.0
    return kp, boxes, real


@pytest.mark.parametrize("criterion", CRITERIA)
def test_each_criterion_alone_rejects_a_window_built_to_fail_it(criterion: str) -> None:
    q = _quality(*_break(criterion))
    failed = [name for name, f in failures(q, THRESHOLDS).items() if f[0]]
    assert failed == [criterion]
    assert REASONS[int(gate_reasons(q, THRESHOLDS)[0])] == criterion


def test_criterion_without_threshold_is_not_applied() -> None:
    q = _quality(*_break("border_fraction"))
    without = {k: v for k, v in THRESHOLDS.items() if k != "border_fraction"}
    assert gate_reasons(q, without).tolist() == [SCORABLE]


def test_box_touching_the_border_within_the_margin() -> None:
    boxes = np.array(
        [[2, 50, 100, 200], [3, 50, 100, 200], [10, 10, 1918, 200], [10, 10, 1917, 1077]],
        np.float32,
    )
    assert touches_border(boxes, SIZE, 2.0).tolist() == [True, False, True, False]


def test_torso_box_ratio_needs_all_four_torso_joints() -> None:
    kp, boxes, _ = _clean_window()
    r = torso_box_ratio(kp[0], boxes[0], 0.1)
    assert np.isfinite(r).all() and (r > 0).all() and (r < 1).all()
    kp[0, 3, HIPS[0], 2] = 0.05
    assert np.isnan(torso_box_ratio(kp[0], boxes[0], 0.1)[3])


def test_interpolated_frames_are_not_real_and_boxes_are_interpolated() -> None:
    frames = np.array([0, 1, 4, 5], np.int64)
    kp = np.zeros((4, 17, 3), np.float32)
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [6, 0, 16, 10], [6, 0, 16, 10]], np.float32)
    real = np.array([True, True, False, True])  # row 2: NaN confidence from the authors
    (t,) = split_trajectories([Trajectory(1, frames, kp, boxes, real)], max_gap=2, min_length=4)
    assert t.real.tolist() == [True, True, False, False, False, True]
    np.testing.assert_allclose(t.boxes[2:4, 0], [2.0, 4.0])


def _treatment(**overrides: object) -> dict[str, object]:
    t: dict[str, object] = {"T": 24, "max_gap": 8, "conf_min": 0.1, "center": "frame"}
    return t | overrides


def test_treated_windows_with_switches_off_match_the_ported_treatment(tmp_path: Path) -> None:
    write_poselift(tmp_path, generate(seed=3, n_train=1, n_test=1))
    vid = official_split(tmp_path)["train"][0]
    video = load_video(tmp_path, "train", vid, CFG)
    x, meta = video_windows(video, 24, 6, 8, 0.1, "frame")
    off = treated_windows(video, _treatment(scale_guard="off", quality_gate="off"), 6, SIZE)
    np.testing.assert_array_equal(off.x, x)
    np.testing.assert_array_equal(off.meta, meta)
    assert (off.reason == SCORABLE).all()


def test_treated_windows_mark_the_reason_with_the_gate_on(tmp_path: Path) -> None:
    write_poselift(tmp_path, generate(seed=3, n_train=1, n_test=1))
    vid = official_split(tmp_path)["train"][0]
    video = load_video(tmp_path, "train", vid, CFG)
    t = _treatment(
        scale_guard="on",
        scale_ratio={"r_med": 0.31, "r_p1": 0.18},
        quality_gate="on",
        quality_thresholds={"box_height": 1e6},  # every window is too small
        border_margin_px=2,
    )
    out = treated_windows(video, t, 6, SIZE)
    assert len(out.x) and (out.reason == CRITERIA.index("box_height") + 1).all()


def test_unquoted_yaml_on_is_refused() -> None:
    with pytest.raises(ValueError, match="quoted"):
        switch({"quality_gate": True}, "quality_gate")
    assert switch({}, "quality_gate") is False
