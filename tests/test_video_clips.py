import numpy as np

from tcc.video.clips import (
    causal_standardise,
    clip_frame_scores,
    clip_starts,
    frame_labels,
    parse_annotations,
    people_crop,
    remove_position_trend,
    remove_scene,
)

# Rows of the official Temporal_Anomaly_Annotation_for_Testing_Videos.txt.
ROWS = """Shoplifting001_x264.mp4  Shoplifting  1550  2000  -1  -1
Shoplifting007_x264.mp4  Shoplifting  550  760  4630  4920
Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1"""


def test_causal_standardise_uses_only_earlier_scores() -> None:
    s = np.array([1.0, 3.0, 2.0, 10.0, 0.0])
    z = causal_standardise(s, 2)
    assert np.isnan(z[:2]).all()
    assert z[2] == 0.0  # mean 2 of the two earlier ones
    assert z[3] == (10.0 - s[:3].mean()) / s[:3].std()
    changed = s.copy()
    changed[4] = 99.0
    assert np.array_equal(causal_standardise(changed, 2)[:4], z[:4], equal_nan=True)


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


def test_people_crop_is_square_inside_the_frame_and_whole_frame_without_people() -> None:
    boxes = np.array([[10, 100, 50, 140, 150], [90, 0, 0, 10, 10]], np.float32)
    x1, y1, x2, y2 = people_crop(boxes, 0, 60, 320, 240, 0.2)
    assert x2 - x1 == y2 - y1 == 140
    assert x1 <= 100 and x2 >= 140 and y1 <= 50 and y2 >= 150
    assert people_crop(boxes, 200, 260, 320, 240, 0.2) == (0, 0, 320, 240)


def test_scene_removal_is_causal_and_cancels_a_constant_scene() -> None:
    x = np.array([[2.0], [2.0], [5.0]], np.float32)
    out = remove_scene(x)
    assert out.tolist() == [[0.0], [0.0], [3.0]]
    changed = x.copy()
    changed[2] = 100.0
    assert remove_scene(changed)[:2].tolist() == out[:2].tolist()


def test_position_trend_removal_flattens_a_ramp() -> None:
    assert np.allclose(remove_position_trend(np.arange(10.0) * 3 + 1), 0.0)
