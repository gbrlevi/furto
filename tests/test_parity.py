"""Pose distribution measures for the parity with PoseLift (session 3B, task 5)."""

import numpy as np

from tcc.data.parity import EDGES, bone_ratios, box_extent_ratio, joint_confidence
from tcc.data.skeleton import HIPS, SHOULDERS

NOSE, ANKLES = 0, [15, 16]


def _pose() -> np.ndarray:
    """Upright pose: nose at y=100, shoulders at 150, hips at 250, ankles at 400."""
    p = np.zeros((17, 3), np.float32)
    p[:, 0] = 500.0
    p[:, 1] = np.linspace(100, 400, 17)
    p[NOSE, 1] = 100.0
    p[SHOULDERS, 0], p[SHOULDERS, 1] = [480.0, 520.0], 150.0
    p[HIPS, 0], p[HIPS, 1] = [485.0, 515.0], 250.0
    p[ANKLES, 1] = 400.0
    p[:, 2] = 0.8
    return p


def test_bone_ratio_is_bone_length_over_torso_length() -> None:
    kp = _pose()[None]
    r = bone_ratios(kp, conf_min=0.1)
    assert r.shape == (1, len(EDGES))
    i = EDGES.index((5, 6))  # shoulder width 40 px, torso 100 px
    assert np.isclose(r[0, i], 0.4)
    doubled = kp.copy()
    doubled[..., :2] *= 2
    np.testing.assert_allclose(bone_ratios(doubled, 0.1), r, rtol=1e-6)


def test_bone_ratio_needs_both_joints_and_the_torso() -> None:
    kp = _pose()[None]
    kp[0, 9, 2] = 0.05  # left wrist
    r = bone_ratios(kp, 0.1)
    assert np.isnan(r[0, EDGES.index((7, 9))]) and np.isfinite(r[0, EDGES.index((5, 7))])
    kp[0, HIPS[1], 2] = 0.0
    assert np.isnan(bone_ratios(kp, 0.1)).all()


def test_box_extent_ratio_uses_confident_keypoints_of_people_with_nose_and_ankles() -> None:
    kp = np.stack([_pose(), _pose(), _pose()])
    boxes = np.array([[450, 90, 550, 410], [450, 50, 550, 450], [0, 0, 1, 1]], np.float32)
    kp[2, ANKLES[0], 2] = 0.05  # third person: one ankle not visible
    r = box_extent_ratio(boxes, kp, conf_min=0.1)
    np.testing.assert_allclose(r[:2], [320 / 300, 400 / 300], rtol=1e-6)
    assert np.isnan(r[2])


def test_joint_confidence_summarizes_each_joint() -> None:
    kp = np.stack([_pose(), _pose()])
    kp[0, 3, 2], kp[1, 3, 2] = 0.2, 0.4
    s = joint_confidence(kp)
    assert set(s) == set(range(17))
    assert np.isclose(s[3]["mean"], 0.3) and np.isclose(s[0]["median"], 0.8)
