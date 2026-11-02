import numpy as np

from tcc.eval.controls import (
    add_noise,
    descriptors,
    hand_to_hip,
    raise_wrist,
    shuffle_joints,
    upside_down,
)


def _windows() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (4, 3, 24, 17)).astype(np.float32)
    x[:, 2] = rng.uniform(0, 1, (4, 24, 17))
    return x


def test_hand_to_hip_reaches_the_hip_and_leaves_the_rest() -> None:
    x = _windows()
    out = hand_to_hip(x, "left", frames=12, ramp=6)
    assert np.allclose(out[:, :2, -6:, 9], x[:, :2, -6:, 11])  # wrist on the hip
    assert np.allclose(out[:, :2, :12, 9], x[:, :2, :12, 9])  # untouched before
    assert np.allclose(out[:, :2, -1, 7], (x[:, :2, -1, 7] + x[:, :2, -1, 11]) / 2)
    others = [v for v in range(17) if v not in (7, 9)]
    assert np.array_equal(out[:, :, :, others], x[:, :, :, others])
    assert np.array_equal(out[:, 2], x[:, 2])


def test_raise_wrist_moves_only_the_last_frames_of_one_wrist() -> None:
    x = _windows()
    out = raise_wrist(x, "right", frames=12, height=2.0)
    assert (out[:, 1, -12:, 10] == -2.0).all() and (out[:, 0, -12:, 10] == 0.0).all()
    out[:, :2, -12:, 10] = x[:, :2, -12:, 10]
    assert np.array_equal(out, x)


def test_flip_noise_and_shuffle_keep_shape_and_confidence() -> None:
    x = _windows()
    rng = np.random.default_rng(1)
    for out in (upside_down(x), add_noise(x, 0.2, rng), shuffle_joints(x, rng)):
        assert out.shape == x.shape
        assert np.allclose(np.sort(out[:, 2], axis=-1), np.sort(x[:, 2], axis=-1))
    assert np.array_equal(upside_down(x)[:, 1], -x[:, 1])


def test_descriptors_of_a_still_pose_and_a_moving_wrist() -> None:
    x = np.zeros((1, 3, 24, 17), np.float32)
    x[:, 2] = 0.5
    still = descriptors(x)
    assert still["pose_speed"][0] == 0.0 and still["confidence"][0] == 0.5
    x[0, 0, :, 9] = np.arange(24) * 0.1  # left wrist 0.1 torso per frame
    moving = descriptors(x)
    assert np.isclose(moving["left_wrist_speed_mean"][0], 0.1)
    assert np.isclose(moving["pose_speed"][0], 0.1 / 17)
    assert moving["right_wrist_speed_max"][0] == 0.0
