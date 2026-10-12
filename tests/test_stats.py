import numpy as np

from tcc.data.schema import VideoPoses
from tcc.data.stats import (
    axis_evidence,
    confidence_histograms,
    frame_counts,
    nan_rows_vs_fractional_boxes,
    nan_runs,
    nan_summary,
    short_trajectory_share,
    track_lengths,
)


def _video(
    frames: list[int],
    tracks: list[int],
    n_frames: int,
    labels: list[int] | None = None,
    keypoints: np.ndarray | None = None,
) -> VideoPoses:
    n = len(frames)
    return VideoPoses(
        video_id="1_1",
        camera="1",
        n_frames=n_frames,
        frame_idx=np.asarray(frames, np.int64),
        track_id=np.asarray(tracks, np.int64),
        bbox=np.tile(np.array([0, 0, 1, 1], np.float32), (n, 1)),
        keypoints=np.ones((n, 17, 3), np.float32) if keypoints is None else keypoints,
        frame_labels=None if labels is None else np.asarray(labels, np.int64),
    )


def test_frame_counts_in_three_conventions() -> None:
    # 5 frames; frame 1 has two people, frame 2 one, frame 4 one; frames 0 and 3 empty
    v = _video([1, 1, 2, 4], [7, 8, 7, 8], n_frames=5, labels=[0, 1, 1, 0, 0])
    counts = frame_counts([v])
    assert counts["all_frames"] == {"normal": 3, "theft": 2}
    assert counts["frames_with_person"] == {"normal": 1, "theft": 2}
    assert counts["person_frames"] == {"normal": 1, "theft": 3}


def test_unlabelled_video_counts_as_normal() -> None:
    counts = frame_counts([_video([0, 2], [1, 1], n_frames=4)])
    assert counts["all_frames"] == {"normal": 4, "theft": 0}
    assert counts["frames_with_person"] == {"normal": 2, "theft": 0}


def test_track_lengths_observed_span_runs_and_gaps() -> None:
    v = _video([0, 1, 2, 5, 6, 3], [1, 1, 1, 1, 1, 2], n_frames=8)
    lengths = track_lengths([v])
    assert sorted(lengths["observed"].tolist()) == [1, 5]
    assert sorted(lengths["span"].tolist()) == [1, 7]
    assert sorted(lengths["runs"].tolist()) == [1, 2, 3]
    assert lengths["gaps"].tolist() == [2]


def test_nan_summary_fractions() -> None:
    kp = np.ones((2, 17, 3), np.float32)
    kp[0, :, 2] = np.nan
    kp[1, 0, 0] = np.nan
    summary = nan_summary(kp)
    assert summary["conf_nan"] == 17
    assert summary["conf_nan_fraction"] == 17 / 34
    assert summary["xy_nan"] == 1
    assert summary["rows_by_conf_nan_count"] == {0: 1, 17: 1}


def test_nan_run_that_is_linear_interpolation_is_detected() -> None:
    kp = np.zeros((5, 17, 3), np.float32)
    kp[:, :, 0] = np.arange(5)[:, None] * 10.0  # x moves 10 px per frame
    kp[:, :, 2] = 0.9
    kp[1:4, :, 2] = np.nan  # frames 1..3 interpolated
    runs = nan_runs(_video([0, 1, 2, 3, 4], [1] * 5, 5, keypoints=kp), tolerance_px=0.5)
    assert runs == [{"length": 3, "interior": True, "max_deviation_px": 0.0, "linear": True}]


def test_nan_run_at_track_edge_is_not_interior() -> None:
    kp = np.ones((3, 17, 3), np.float32)
    kp[2, :, 2] = np.nan
    runs = nan_runs(_video([0, 1, 2], [1] * 3, 3, keypoints=kp), tolerance_px=0.5)
    assert runs == [{"length": 1, "interior": False}]


def test_nan_run_off_the_line_is_not_linear() -> None:
    kp = np.zeros((3, 17, 3), np.float32)
    kp[:, :, 2] = 0.9
    kp[1, :, 2] = np.nan
    kp[1, :, 1] = 7.0  # middle frame 7 px away from the straight line
    runs = nan_runs(_video([0, 1, 2], [1] * 3, 3, keypoints=kp), tolerance_px=0.5)
    assert runs[0]["max_deviation_px"] == 7.0 and runs[0]["linear"] is False


def test_nan_rows_vs_fractional_boxes_counts_the_overlap() -> None:
    kp = np.ones((3, 17, 3), np.float32)
    kp[1:, :, 2] = np.nan
    v = _video([0, 1, 2], [1] * 3, 3, keypoints=kp)
    v.bbox[1] = [0.5, 0, 1, 1]
    v.bbox[0] = [0, 0, 1.5, 1]
    assert nan_rows_vs_fractional_boxes([v]) == {
        "all_nan_rows": 2,
        "fractional_box_rows": 2,
        "both": 1,
    }


def test_axis_evidence_finds_the_vertical_column() -> None:
    kp = np.zeros((1, 17, 3), np.float32)
    kp[0, :, 0] = 500.0  # column 0 is horizontal here
    kp[0, 0, 1], kp[0, 15, 1], kp[0, 16, 1] = 100.0, 300.0, 300.0  # column 1 vertical
    kp[0, 5, 0], kp[0, 6, 0] = 480.0, 520.0
    ev = axis_evidence(kp, np.array([[470, 90, 530, 310]], np.float32))
    assert ev["mean_nose_minus_ankles_per_column"] == [0.0, -200.0]
    assert ev["mean_shoulder_gap_per_column"] == [40.0, 0.0]
    assert ev["columns_xy"]["median_x2_minus_max_x_if_xyxy"] == 10.0


def test_confidence_histograms_skip_nan() -> None:
    kp = np.zeros((3, 17, 3), np.float32)
    kp[:, :, 2] = [[0.1], [0.9], [np.nan]]
    hist = confidence_histograms(kp, np.array([0.0, 0.5, 1.0]))
    assert hist.shape == (17, 2)
    assert hist[0].tolist() == [1, 1]


def test_short_trajectory_share_counts_rows_and_frames() -> None:
    # track 1: frames 0..9 (span 10, kept with min_length 10);
    # track 2: frames 0..3 then 20..22 (gap > max_gap: two short pieces)
    frames = list(range(10)) + [0, 1, 2, 3, 20, 21, 22]
    tracks_ = [1] * 10 + [2] * 7
    share = short_trajectory_share([_video(frames, tracks_, n_frames=30)], 2, 10)
    assert share == {
        "pieces": 3,
        "short_pieces": 2,
        "rows": 17,
        "short_rows": 7,
        "frames": 13,
        "frames_lost": 3,
    }
