"""Evaluation harness (ROADMAP stage 4; session 3, tasks B2 and B3)."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from tcc.eval.events import event_metrics, label_intervals
from tcc.eval.frame_scores import fill, most_normal, post_process, smooth, window_frame_scores
from tcc.eval.metrics import auc_pr, auc_roc, eer, threshold_metrics
from tcc.eval.threshold import (
    alert_rate_curve,
    alerts_per_hour,
    candidate_thresholds,
    quantile_threshold,
    threshold_for_rate,
)

# Declared tolerance of the random-score check: with 20,000 frames the standard error of
# the AUC is about 0.004, so 0.02 is five standard errors.
RANDOM_AUC_TOLERANCE = 0.02


def _labels(rng: np.random.Generator, n: int = 20_000) -> np.ndarray:
    return (rng.random(n) < 0.4).astype(np.int64)


def test_random_score_gives_auc_close_to_one_half() -> None:
    rng = np.random.default_rng(0)
    y = _labels(rng)
    s = rng.random(len(y))
    assert abs(auc_roc(y, s) - 0.5) < RANDOM_AUC_TOLERANCE


def test_oracle_score_gives_auc_one_and_eer_zero() -> None:
    y = _labels(np.random.default_rng(1))
    s = y.astype(np.float64)
    assert auc_roc(y, s) == 1.0
    assert auc_pr(y, s, "theft", "trapezoid") == 1.0
    assert auc_pr(y, s, "normal", "trapezoid") == 1.0
    assert eer(y, s) == 0.0


def test_random_auc_pr_is_the_positive_class_share_for_each_positive_class() -> None:
    rng = np.random.default_rng(2)
    y = _labels(rng)
    s = rng.random(len(y))
    assert abs(auc_pr(y, s, "theft", "trapezoid") - y.mean()) < RANDOM_AUC_TOLERANCE
    assert abs(auc_pr(y, s, "normal", "trapezoid") - (1 - y.mean())) < RANDOM_AUC_TOLERANCE
    assert abs(auc_pr(y, s, "theft", "average_precision") - y.mean()) < RANDOM_AUC_TOLERANCE


def test_normal_positive_auc_pr_uses_normal_labels_and_negated_scores() -> None:
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.2, 0.8, 0.7, 0.6, 0.3])
    assert auc_pr(y, s, "normal", "trapezoid") == auc_pr(1 - y, -s, "theft", "trapezoid")
    with pytest.raises(ValueError):
        auc_pr(y, s, "both", "trapezoid")


def test_eer_where_false_positive_and_false_negative_rates_meet() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    assert eer(y, s) == 0.25


def test_threshold_metrics_on_a_known_case() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.3, 0.6])
    m = threshold_metrics(y, s, 0.5)
    assert m["confusion"] == {"tp": 3, "fp": 1, "fn": 1, "tn": 3}
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["specificity"] == 0.75
    assert m["f1"] == pytest.approx(0.75)
    assert m["h_prs"] == pytest.approx(0.75)


def test_threshold_metrics_count_only_scores_strictly_above_the_threshold() -> None:
    m = threshold_metrics(np.array([1, 0]), np.array([0.5, 0.5]), 0.5)
    assert m["confusion"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 1}
    assert m["h_prs"] == 0.0


def test_h_prs_is_the_harmonic_mean_of_precision_recall_and_specificity() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2, 0.2, 0.2])
    m = threshold_metrics(y, s, 0.5)
    p, r, sp = 0.5, 0.5, 0.75
    assert m["h_prs"] == pytest.approx(3 / (1 / p + 1 / r + 1 / sp))


# Window -> frame attribution. meta rows are [video, track, first frame].
META = np.array([[0, 1, 0], [0, 1, 1], [0, 2, 0]], np.int64)
SCORES = np.array([0.2, 0.5, 0.3])


def test_center_mode_puts_each_window_score_on_first_frame_plus_half_t() -> None:
    out = window_frame_scores(META, SCORES, 4, 8, "center")
    assert np.isnan(out[[0, 1, 4, 5, 6, 7]]).all()
    assert out[2] == 0.3 and out[3] == 0.5  # frame 2: max of track 1 (0.2) and track 2 (0.3)


def test_last_mode_puts_each_window_score_on_its_last_frame() -> None:
    out = window_frame_scores(META, SCORES, 4, 8, "last")
    assert out[3] == 0.3 and out[4] == 0.5
    assert np.isnan(out[[0, 1, 2, 5, 6, 7]]).all()


def test_max_cover_mode_takes_the_largest_window_covering_each_frame() -> None:
    out = window_frame_scores(META, SCORES, 4, 8, "max_cover")
    np.testing.assert_array_equal(out[:5], [0.3, 0.5, 0.5, 0.5, 0.5])
    assert np.isnan(out[5:]).all()


def test_frame_score_is_the_maximum_over_people() -> None:
    meta = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]], np.int64)
    out = window_frame_scores(meta, np.array([0.4, 0.9, 0.1]), 4, 4, "center")
    assert out[2] == 0.9


def test_not_scorable_windows_leave_the_frame_most_normal() -> None:
    """ADR-0014: a frame with only filtered windows is scored like a frame without people."""
    meta = np.array([[0, 1, 0], [0, 2, 1]], np.int64)
    raw = window_frame_scores(meta, np.array([np.nan, 0.7]), 4, 6, "center")
    assert np.isnan(raw[2]) and raw[3] == 0.7
    other = np.array([0.2, np.nan, 0.05])
    floor = most_normal([raw, other])
    assert floor == 0.05
    filled = fill(raw, floor)
    assert filled[2] == 0.05 and filled[0] == 0.05 and filled[3] == 0.7


def test_most_normal_is_the_smallest_finite_score_of_all_videos() -> None:
    assert most_normal([np.array([np.nan, 3.0]), np.array([2.0, np.nan])]) == 2.0
    with pytest.raises(ValueError):
        most_normal([np.array([np.nan])])


def test_smoothing_applies_each_sigma_in_sequence() -> None:
    s = np.random.default_rng(3).random(200)
    expected = s.copy()
    for sigma in range(1, 7):
        expected = gaussian_filter1d(expected, sigma)
    np.testing.assert_array_equal(smooth(s, [1, 2, 3, 4, 5, 6]), expected)
    np.testing.assert_array_equal(smooth(s, []), s)


def test_post_process_fills_then_smooths_each_video() -> None:
    raw = [np.array([np.nan, 1.0, 2.0, np.nan]), np.array([np.nan, 0.5])]
    out = post_process(raw, 0.5, [2.0])
    np.testing.assert_array_equal(out[0], gaussian_filter1d(np.array([0.5, 1, 2, 0.5]), 2.0))
    np.testing.assert_array_equal(out[1], gaussian_filter1d(np.array([0.5, 0.5]), 2.0))


def test_label_intervals_are_half_open_runs_of_theft() -> None:
    assert label_intervals(np.array([0, 1, 1, 0, 0, 1])) == [(1, 3), (5, 6)]
    assert label_intervals(np.array([0, 0])) == []


def test_event_metrics_match_alerts_to_theft_events_with_tolerance() -> None:
    labels = [np.array([0] * 10 + [1] * 10 + [0] * 30 + [1] * 5 + [0] * 45)]
    alerts = [[(22, 25), (60, 70), (80, 90)]]  # 22 is within 3 frames of the first event
    m = event_metrics(labels, alerts, tolerance=3)
    assert m == {
        "events": 2,
        "events_detected": 1,
        "event_recall": 0.5,
        "alerts": 3,
        "false_alerts": 2,
    }
    assert event_metrics(labels, alerts, tolerance=0)["events_detected"] == 0


def _video(scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, np.float64)
    return s, np.isfinite(s)


def test_alerts_per_hour_uses_the_alert_logic_and_all_frames() -> None:
    # 2 alerts of >= 3 frames (the 2-frame run is too short) in 3600 frames at 1 fps.
    s = np.zeros(3600)
    s[[10, 11, 12, 100, 101, 200, 201, 202, 203]] = 1.0
    rate = alerts_per_hour([(s, np.ones(3600, np.bool_))], 0.5, 3, 5, fps=1)
    assert rate == 2.0


def test_threshold_for_rate_is_the_lowest_candidate_within_the_target() -> None:
    s = np.zeros(7200)
    s[0:10], s[100:110], s[200:210] = 0.9, 0.6, 0.3  # three 10-frame runs, 2 h at 1 fps
    videos = [(s, np.ones(len(s), np.bool_))]
    candidates = np.array([0.1, 0.4, 0.7, 0.95])
    np.testing.assert_array_equal(
        alert_rate_curve(videos, candidates, 5, 1, fps=1), [1.5, 1.0, 0.5, 0.0]
    )
    assert threshold_for_rate(videos, 1.0, candidates, 5, 1, fps=1) == (0.4, True)
    assert threshold_for_rate(videos, 0.0, candidates, 5, 1, fps=1) == (0.95, True)
    assert threshold_for_rate(videos, -1.0, candidates, 5, 1, fps=1) == (0.95, False)


def test_candidate_thresholds_are_unique_quantiles_of_the_sample() -> None:
    sample = np.arange(101, dtype=np.float64)
    np.testing.assert_array_equal(candidate_thresholds(sample, 0.5, 1.0, 3), [50, 75, 100])
    assert quantile_threshold(np.array([0.0, np.nan, 10.0]), 0.5) == 5.0
