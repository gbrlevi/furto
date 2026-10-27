"""Bootstrap confidence interval by video (session 3, task B4)."""

import numpy as np
from sklearn.metrics import roc_auc_score

from tcc.eval.bootstrap import bootstrap_auc_by_video
from tcc.eval.metrics import auc_roc


def _correlated_videos(
    rng: np.random.Generator, n_videos: int = 30, frames: int = 200
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Each video has its own offset shared by all its frames: frames of one video are
    strongly correlated, as in real video."""
    ys, ss = [], []
    for _ in range(n_videos):
        y = np.zeros(frames, np.int64)
        y[rng.integers(0, frames // 2) :][: frames // 3] = 1
        ss.append(y * 0.5 + rng.normal(0, 1.0) + rng.normal(0, 0.3, frames))
        ys.append(y)
    return ys, ss


def test_bootstrap_by_video_is_reproducible_and_contains_the_estimate() -> None:
    ys, ss = _correlated_videos(np.random.default_rng(4))
    a = bootstrap_auc_by_video(ys, ss, n_resamples=500, seed=7, level=0.95)
    b = bootstrap_auc_by_video(ys, ss, n_resamples=500, seed=7, level=0.95)
    assert a == b
    point = auc_roc(np.concatenate(ys), np.concatenate(ss))
    assert a["low"] < point < a["high"]
    assert a["resamples_used"] + a["resamples_one_class"] == 500


def test_bootstrap_by_video_is_wider_than_by_frame_on_correlated_frames() -> None:
    """Why the CI resamples videos (session 3, B4): resampling frames ignores the
    correlation inside a video and narrows the interval."""
    rng = np.random.default_rng(5)
    ys, ss = _correlated_videos(rng)
    by_video = bootstrap_auc_by_video(ys, ss, n_resamples=500, seed=0, level=0.95)
    y, s = np.concatenate(ys), np.concatenate(ss)
    frame_rng = np.random.default_rng(0)
    aucs = []
    for _ in range(500):
        i = frame_rng.integers(0, len(y), len(y))
        aucs.append(roc_auc_score(y[i], s[i]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    assert by_video["high"] - by_video["low"] > 1.25 * (high - low)


def test_bootstrap_skips_resamples_with_a_single_class() -> None:
    ys = [np.zeros(10, np.int64), np.ones(10, np.int64)]
    ss = [np.zeros(10), np.ones(10)]
    out = bootstrap_auc_by_video(ys, ss, n_resamples=200, seed=0, level=0.95)
    assert out["resamples_one_class"] > 0
    assert out["resamples_used"] + out["resamples_one_class"] == 200
