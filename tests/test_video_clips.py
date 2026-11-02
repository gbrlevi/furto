import numpy as np

from tcc.video.clips import clip_frame_scores, clip_starts, frame_labels, parse_annotations

# Rows of the official Temporal_Anomaly_Annotation_for_Testing_Videos.txt.
ROWS = """Shoplifting001_x264.mp4  Shoplifting  1550  2000  -1  -1
Shoplifting007_x264.mp4  Shoplifting  550  760  4630  4920
Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1"""


def test_annotations_keep_one_or_two_intervals_of_the_category() -> None:
    spans = parse_annotations(ROWS, "Shoplifting")
    assert spans == {
        "Shoplifting001_x264": [(1550, 2000)],
        "Shoplifting007_x264": [(550, 760), (4630, 4920)],
    }


def test_frame_labels_mark_only_the_interval() -> None:
    y = frame_labels(6, [(2, 4)])
    assert y.tolist() == [0, 0, 1, 1, 0, 0]


def test_clip_starts_use_only_complete_clips() -> None:
    assert clip_starts(100, 61, 16).tolist() == [0, 16, 32]
    assert clip_starts(10, 61, 16).tolist() == []


def test_clip_scores_are_causal_and_cover_until_the_next_clip() -> None:
    s = clip_frame_scores(np.array([0, 2]), np.array([1.0, 2.0]), 3, 7)
    assert np.isnan(s[:2]).all()  # no clip has ended yet
    assert s[2:].tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]
