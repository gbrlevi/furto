"""Pure parts of the pose extraction (the models themselves run only in .venv-pose)."""

import numpy as np
import pytest

from tcc.pipeline.pose_extract import Rows
from tcc.pipeline.resample import frames_to_keep


def test_30_to_15_fps_keeps_every_other_frame() -> None:
    assert frames_to_keep(30.0, 10, 15.0).tolist() == [0, 2, 4, 6, 8]


def test_resampling_keeps_the_duration() -> None:
    idx = frames_to_keep(30.0, 9001, 15.0)
    assert len(idx) == 4500  # 300.03 s at 15 fps
    assert idx[-1] < 9001


def test_non_integer_ratio_takes_nearest_frames() -> None:
    assert frames_to_keep(25.0, 10, 15.0).tolist() == [0, 2, 3, 5, 7, 8]


def test_same_rate_keeps_every_frame() -> None:
    assert frames_to_keep(15.0, 5, 15.0).tolist() == [0, 1, 2, 3, 4]


def test_upsampling_is_refused() -> None:
    with pytest.raises(ValueError, match="upsample"):
        frames_to_keep(10.0, 5, 15.0)


def test_rows_become_contract_records_in_xyc() -> None:
    rows = Rows()
    tracks = np.array(
        [[10, 20, 50, 120, 7, 0.9, 0, 0], [60, 20, 90, 110, 3, 0.8, 0, 1]], np.float32
    )
    xy = np.random.default_rng(0).uniform(10, 100, (2, 17, 2)).astype(np.float32)
    scores = np.full((2, 17), 0.7, np.float32)
    rows.add(4, tracks, xy, scores)
    video = rows.to_video("G421", "G421", n_frames=10)
    assert video.track_id.tolist() == [7, 3]
    assert video.frame_idx.tolist() == [4, 4]
    np.testing.assert_array_equal(video.keypoints[1, :, :2], xy[1])
    np.testing.assert_array_equal(video.keypoints[:, :, 2], scores)
    assert next(video.records()).bbox == (10.0, 20.0, 50.0, 120.0)


def test_no_people_gives_an_empty_video() -> None:
    video = Rows().to_video("x", "x", n_frames=3)
    assert len(video) == 0 and video.n_frames == 3
