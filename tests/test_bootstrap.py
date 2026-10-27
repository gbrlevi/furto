"""Bootstrap confidence interval by video (session 3, task B4; P17 item 17)."""

import numpy as np
from sklearn.metrics import roc_auc_score

from tcc.eval.bootstrap import bootstrap_by_video
from tcc.eval.metrics import auc_roc


def _correlated_videos(
    rng: np.random.Generator, n_videos: int = 30, frames: int = 200, theft_every: int = 1
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Each video has its own offset shared by all its frames: frames of one video are
    strongly correlated, as in real video. Every `theft_every`-th video has theft."""
    ys, ss = [], []
    for v in range(n_videos):
        y = np.zeros(frames, np.int64)
        if v % theft_every == 0:
            y[rng.integers(0, frames // 2) :][: frames // 3] = 1
        ss.append(y * 0.5 + rng.normal(0, 1.0) + rng.normal(0, 0.3, frames))
        ys.append(y)
    return ys, ss


def test_bootstrap_by_video_is_reproducible_and_contains_the_estimate() -> None:
    ys, ss = _correlated_videos(np.random.default_rng(4), theft_every=2)
    a = bootstrap_by_video(ys, [ss], auc_roc, n_resamples=500, seed=7, level=0.95)
    b = bootstrap_by_video(ys, [ss], auc_roc, n_resamples=500, seed=7, level=0.95)
    assert a == b
    point = auc_roc(np.concatenate(ys), np.concatenate(ss))
    assert a["low"] < point < a["high"]
    assert a["resamples"] == 500


def test_bootstrap_by_video_is_wider_than_by_frame_on_correlated_frames() -> None:
    """Why the CI resamples videos (session 3, B4): resampling frames ignores the
    correlation inside a video and narrows the interval."""
    ys, ss = _correlated_videos(np.random.default_rng(5))
    by_video = bootstrap_by_video(ys, [ss], auc_roc, n_resamples=500, seed=0, level=0.95)
    y, s = np.concatenate(ys), np.concatenate(ss)
    frame_rng = np.random.default_rng(0)
    aucs = []
    for _ in range(500):
        i = frame_rng.integers(0, len(y), len(y))
        aucs.append(roc_auc_score(y[i], s[i]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    assert by_video["high"] - by_video["low"] > 1.25 * (high - low)


def test_strata_keep_their_size_so_every_resample_has_both_classes() -> None:
    """Theft videos and normal-only videos are resampled separately (P17 item 17)."""
    ys = [np.zeros(10, np.int64)] * 9 + [np.r_[np.zeros(5), np.ones(5)].astype(np.int64)]
    ss = [np.random.default_rng(i).random(10) for i in range(10)]
    out = bootstrap_by_video(ys, [ss], auc_roc, n_resamples=300, seed=0, level=0.95)
    assert out["resamples"] == 300
    assert out["strata"] == {"theft_videos": 1, "normal_videos": 9}


def test_interval_is_of_the_mean_over_seeds_on_the_same_resampled_videos() -> None:
    ys, a = _correlated_videos(np.random.default_rng(6), theft_every=2)
    b = [s + np.random.default_rng(i).normal(0, 0.5, len(s)) for i, s in enumerate(a)]
    both = bootstrap_by_video(ys, [a, b], auc_roc, n_resamples=400, seed=1, level=0.95)
    y = np.concatenate(ys)
    mean = (auc_roc(y, np.concatenate(a)) + auc_roc(y, np.concatenate(b))) / 2
    assert both["low"] < mean < both["high"]
    assert both["seeds"] == 2
