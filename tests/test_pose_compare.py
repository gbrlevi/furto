"""Pose comparison used by the test-time flip rule and the optimized HRNet (session 3B)."""

import numpy as np

from tcc.data.skeleton import HIPS, SHOULDERS
from tcc.pipeline.pose_compare import displacement_summary, normalized_displacement


def _pose(torso: float = 100.0) -> np.ndarray:
    """Standing pose with mid-shoulder 100 px above mid-hip by default."""
    p = np.zeros((17, 3), np.float32)
    p[:, 0] = np.linspace(400, 440, 17)
    p[:, 1] = 300.0
    p[SHOULDERS, 0], p[HIPS, 0] = [400.0, 440.0], [400.0, 440.0]
    p[SHOULDERS, 1] = 300.0
    p[HIPS, 1] = 300.0 + torso
    p[:, 2] = 0.9
    return p


def test_displacement_is_divided_by_the_torso_of_the_reference_pose() -> None:
    a = _pose()[None]
    b = a.copy()
    b[0, 9, 0] += 3.0  # left wrist 3 px to the right
    b[0, 10, 1] += 4.0  # right wrist 4 px down
    d = normalized_displacement(a, b, conf_min=0.1)
    assert d.shape == (1, 17)
    assert d[0, 9] == np.float32(0.03) and d[0, 10] == np.float32(0.04)
    assert (d[0, :9] == 0).all()


def test_doubling_the_image_keeps_the_normalized_displacement() -> None:
    a, b = _pose()[None], _pose()[None]
    b[0, 0, 0] += 5.0
    a2, b2 = a.copy(), b.copy()
    a2[..., :2] *= 2
    b2[..., :2] *= 2
    np.testing.assert_allclose(
        normalized_displacement(a, b, 0.1), normalized_displacement(a2, b2, 0.1), rtol=1e-6
    )


def test_joints_below_the_confidence_in_either_pose_are_left_out() -> None:
    a, b = _pose()[None], _pose()[None]
    a[0, 3, 2] = 0.05
    b[0, 4, 2] = 0.05
    d = normalized_displacement(a, b, conf_min=0.1)
    assert np.isnan(d[0, 3]) and np.isnan(d[0, 4]) and np.isfinite(d[0, 5])


def test_pose_without_a_valid_torso_has_no_displacement() -> None:
    a, b = _pose()[None], _pose()[None]
    a[0, HIPS[0], 2] = 0.0
    assert np.isnan(normalized_displacement(a, b, 0.1)).all()


def test_summary_counts_points_and_poses_without_torso() -> None:
    a = np.stack([_pose(), _pose()])
    b = a.copy()
    b[0, :, 0] += 2.0  # every joint 2 px: 0.02 of the torso
    a[1, SHOULDERS[0], 2] = 0.0
    s = displacement_summary(normalized_displacement(a, b, 0.1))
    assert s["points"] == 17 and s["poses"] == 2 and s["poses_without_torso"] == 1
    assert abs(s["mean"] - 0.02) < 1e-6 and abs(s["median"] - 0.02) < 1e-6
