from pathlib import Path

import numpy as np
import pytest

from tcc.pipeline.scoring import (
    alert_intervals,
    frame_max,
    last_frame_scores,
    load_extracted,
    overlaps,
    quantile_threshold,
    source_interval_to_output,
)


def test_window_score_goes_to_its_last_frame() -> None:
    meta = np.array([[0, 7, 0], [0, 7, 1], [0, 3, 0]], np.int64)
    scores = np.array([0.5, 0.9, 0.2], np.float32)
    out = last_frame_scores(meta, scores, length=24)
    assert out == {(7, 23): 0.5, (7, 24): pytest.approx(0.9), (3, 23): pytest.approx(0.2)}


def test_nan_window_scores_are_skipped() -> None:
    meta = np.array([[0, 1, 0]], np.int64)
    assert last_frame_scores(meta, np.array([np.nan], np.float32), 24) == {}


def test_frame_score_is_the_maximum_over_people() -> None:
    per_person = {(7, 2): 0.5, (3, 2): 0.8, (7, 3): 0.1}
    out = frame_max(per_person, n_frames=5)
    assert np.isnan(out[[0, 1, 4]]).all()
    assert out[2] == 0.8 and out[3] == 0.1


def test_quantile_threshold_ignores_nan() -> None:
    scores = np.array([np.nan, *np.arange(1, 101)], np.float32)
    assert quantile_threshold(scores, 0.5) == pytest.approx(50.5)
    with pytest.raises(ValueError):
        quantile_threshold(np.array([np.nan], np.float32), 0.5)


def test_alerts_are_runs_strictly_above_threshold() -> None:
    s = np.array([0.1, 0.6, 0.7, np.nan, 0.9, 0.5, 0.95])
    assert alert_intervals(s, 0.5) == [(1, 3), (4, 5), (6, 7)]


def test_interval_overlap_is_half_open() -> None:
    assert overlaps((0, 5), (4, 9))
    assert not overlaps((0, 5), (5, 9))


def test_annotation_in_source_frames_maps_to_output_frames() -> None:
    keep = np.arange(0, 100, 2)  # 30 -> 15 fps
    assert source_interval_to_output(keep, 10, 20) == (5, 11)
    assert source_interval_to_output(keep, 201, 300) is None


def test_load_extracted_builds_the_contract(tmp_path: Path) -> None:
    path = tmp_path / "v_poses.npz"
    np.savez_compressed(
        path,
        frame_idx=np.array([0, 1]),
        track_id=np.array([4, 4]),
        bbox=np.array([[0, 0, 10, 20], [1, 1, 11, 21]], np.float32),
        keypoints=np.ones((2, 17, 3), np.float32),
        n_frames=3,
    )
    v = load_extracted(path, "v", "cam")
    assert v.n_frames == 3 and len(v) == 2 and v.frame_labels is None
