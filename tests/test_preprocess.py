"""Acceptance tests of ROADMAP stage 3 (treatment)."""

import numpy as np
import pytest

from tcc.data.schema import VideoPoses
from tcc.data.skeleton import HIPS, SHOULDERS
from tcc.preprocess.normalize import normalize
from tcc.preprocess.occlusion import drop_joints
from tcc.preprocess.trajectories import Trajectory, split_trajectories, tracks
from tcc.preprocess.windows import (
    TheftInTrainingError,
    check_no_theft,
    make_windows,
    window_starts,
)
from tests.fixtures.synthetic import generate


def _pose_sequence(n: int, seed: int = 0) -> np.ndarray:
    _, seq = generate(seed=seed, n_train=1, n_test=0)[0].people[1]
    assert len(seq) >= n
    return seq[:n].copy()


def _track(frames: list[int]) -> Trajectory:
    kp = np.zeros((len(frames), 17, 3), np.float32)
    kp[:, :, 0] = np.asarray(frames, np.float32)[:, None] * 2.0  # x moves 2 px per frame
    kp[:, :, 2] = 0.9
    return Trajectory(1, np.asarray(frames, np.int64), kp)


def test_tracks_group_rows_by_person_in_order_of_first_appearance() -> None:
    v = VideoPoses(
        video_id="1_1",
        camera="1",
        n_frames=3,
        frame_idx=np.array([0, 0, 1, 2], np.int64),
        track_id=np.array([9, 4, 4, 9], np.int64),
        bbox=np.zeros((4, 4), np.float32),
        keypoints=np.zeros((4, 17, 3), np.float32),
        frame_labels=None,
    )
    out = tracks(v)
    assert [t.track_id for t in out] == [9, 4]
    assert out[0].frames.tolist() == [0, 2] and out[1].frames.tolist() == [0, 1]


def test_gap_up_to_k_frames_is_interpolated() -> None:
    k = 3
    frames = list(range(10)) + list(range(10 + k, 30))  # k missing frames
    (traj,) = split_trajectories([_track(frames)], max_gap=k, min_length=5)
    assert traj.frames.tolist() == list(range(30))
    np.testing.assert_allclose(traj.keypoints[10:13, 0, 0], [20.0, 22.0, 24.0])


def test_gap_longer_than_k_frames_splits_the_track() -> None:
    k = 3
    frames = list(range(10)) + list(range(10 + k + 1, 30))  # k + 1 missing frames
    out = split_trajectories([_track(frames)], max_gap=k, min_length=5)
    assert [t.frames.tolist() for t in out] == [list(range(10)), list(range(14, 30))]


def test_trajectories_shorter_than_t_are_discarded() -> None:
    out = split_trajectories([_track(list(range(23))), _track(list(range(24)))], 8, 24)
    assert [len(t.frames) for t in out] == [24]


def test_points_below_confidence_threshold_are_zeroed() -> None:
    j = _pose_sequence(24)[None]
    j[0, 5, 7, 2] = 0.05
    out = normalize(j, conf_min=0.1, center="frame")
    assert out[0, 5, 7].tolist() == [0.0, 0.0, 0.0]
    assert out[0, 5, 8, 2] == j[0, 5, 8, 2]


def test_normalization_is_invariant_to_translation_and_scale() -> None:
    j = _pose_sequence(24)[None]
    moved = j.copy()
    moved[..., :2] = moved[..., :2] * 2.5 + np.array([300.0, -120.0], np.float32)
    for center in ["frame", "window"]:
        a = normalize(j, 0.1, center)
        b = normalize(moved, 0.1, center)
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_normalization_centres_on_the_hips_and_scales_by_the_torso() -> None:
    j = _pose_sequence(24)[None]
    j[..., 2] = 0.9
    out = normalize(j, 0.1, "frame")
    frames = out[0]
    hips = frames[:, HIPS, :2].mean(1)
    torso = np.linalg.norm(frames[:, SHOULDERS, :2].mean(1) - hips, axis=-1)
    np.testing.assert_allclose(hips, 0.0, atol=1e-5)
    assert torso.mean() == pytest.approx(1.0, rel=1e-5)


def test_training_windows_use_stride_and_always_reach_the_end() -> None:
    assert window_starts(40, 24, 6) == [0, 6, 12, 16]
    assert window_starts(36, 24, 6) == [0, 6, 12]
    assert window_starts(24, 24, 6) == [0]


def test_evaluation_windows_cover_every_start() -> None:
    assert window_starts(30, 24, 1) == list(range(7))


def test_make_windows_layout_and_metadata() -> None:
    traj = Trajectory(7, np.arange(100, 130), _pose_sequence(30))
    x, meta = make_windows([traj], length=24, stride=6, conf_min=0.1, center="frame", video_idx=2)
    assert x.shape == (2, 3, 24, 17) and x.dtype == np.float32
    assert meta.tolist() == [[2, 7, 100], [2, 7, 106]]
    expected = normalize(traj.keypoints[6:30][None], 0.1, "frame")[0].transpose(2, 0, 1)
    np.testing.assert_array_equal(x[1], expected)


def test_training_window_touching_theft_is_an_error() -> None:
    labels = np.zeros(60, np.int64)
    labels[40] = 1
    meta = np.array([[0, 1, 0], [0, 1, 20]], np.int64)
    with pytest.raises(TheftInTrainingError):
        check_no_theft(meta, labels, 24)
    check_no_theft(meta[:1], labels, 24)
    check_no_theft(meta, None, 24)


def test_drop_joints_only_zeroes_confidence() -> None:
    traj = Trajectory(1, np.arange(200), np.ones((200, 17, 3), np.float32))
    (out,) = drop_joints([traj], 0.3, np.random.default_rng(0))
    dropped = out.keypoints[..., 2] == 0
    assert 0.25 < dropped.mean() < 0.35
    np.testing.assert_array_equal(out.keypoints[..., :2], traj.keypoints[..., :2])
    assert (traj.keypoints[..., 2] == 1).all()
