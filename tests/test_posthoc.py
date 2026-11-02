import numpy as np
from sklearn.metrics import roc_auc_score

from tcc.eval.posthoc import (
    auc,
    events_with_room,
    people_per_frame,
    within_video_auc,
    within_video_null,
)


def test_rank_auc_equals_sklearn_with_ties() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.integers(0, 10, 500).astype(float)  # many ties
    assert np.isclose(auc(y, s), roc_auc_score(y, s), rtol=0, atol=1e-12)
    assert np.isnan(auc(np.zeros(5, np.int64), s[:5]))


def test_people_per_frame_counts_rows() -> None:
    assert people_per_frame(np.array([0, 0, 2, 2, 2]), 4).tolist() == [2, 0, 3, 0]


def test_within_video_auc_skips_one_class_videos_and_respects_the_mask() -> None:
    y1, s1 = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.9, 0.8])
    y2, s2 = np.zeros(4, np.int64), np.arange(4.0)
    mask = np.array([True, True, True, False])
    value, videos = within_video_auc([y1, y2], [s1, s2], [mask, np.ones(4, bool)])
    assert (value, videos) == (1.0, 1)


def test_null_of_a_perfect_score_stays_below_it() -> None:
    labels = [np.r_[np.zeros(40), np.ones(10), np.zeros(50)].astype(np.int64) for _ in range(5)]
    scores = [y.astype(float) for y in labels]
    masks = [np.ones(100, bool) for _ in labels]
    null = within_video_null(labels, scores, masks, 200, 0)
    assert within_video_auc(labels, scores, masks)[0] == 1.0
    assert null.max() < 1.0
    assert np.array_equal(null, within_video_null(labels, scores, masks, 200, 0))


def test_events_with_room_needs_a_long_enough_scored_run() -> None:
    y = np.r_[np.zeros(10), np.ones(5), np.zeros(30), np.ones(5), np.zeros(5)].astype(np.int64)
    scored = np.ones(len(y), bool)
    scored[40:] = False  # second event never scored
    assert events_with_room(y, scored, min_run=15, after=10) == 1
    assert events_with_room(y, scored, min_run=16, after=10) == 0
