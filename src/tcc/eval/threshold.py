"""Threshold from validation scores only (rule 2) and the alerts-per-hour simulation.

Human-review zone (CLAUDE.md §6). The rate uses the alert logic of `tcc.eval.alerts`
(minimum duration and merging, session 2D) over every frame of the videos.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from tcc.eval.alerts import alerts

FloatArray = npt.NDArray[np.float64]
Video = tuple[FloatArray, npt.NDArray[np.bool_]]  # frame score and scored-frame mask


def quantile_threshold(sample: FloatArray, q: float) -> float:
    """Quantile `q` of the finite values of a validation sample."""
    finite = sample[np.isfinite(sample)]
    if len(finite) == 0:
        raise ValueError("no finite score to take a quantile from")
    return float(np.quantile(finite, q))


def alerts_per_hour(
    videos: Sequence[Video],
    threshold: float,
    min_alert_frames: int,
    merge_gap_frames: int,
    fps: float,
) -> float:
    """Alerts of all videos divided by their total duration (all frames) in hours."""
    n = sum(len(alerts(s, m, threshold, min_alert_frames, merge_gap_frames)) for s, m in videos)
    hours = sum(len(s) for s, _ in videos) / fps / 3600
    return n / hours if hours > 0 else float("nan")


def candidate_thresholds(sample: FloatArray, low: float, high: float, n: int) -> FloatArray:
    """Unique quantiles of the finite sample at `n` evenly spaced levels from `low` to
    `high` (the original PoC used 0.5, 0.99999 and 400)."""
    finite = sample[np.isfinite(sample)]
    return np.unique(np.quantile(finite, np.linspace(low, high, n)))


def alert_rate_curve(
    videos: Sequence[Video],
    candidates: FloatArray,
    min_alert_frames: int,
    merge_gap_frames: int,
    fps: float,
) -> FloatArray:
    """Alerts per hour at each candidate threshold."""
    return np.array(
        [alerts_per_hour(videos, t, min_alert_frames, merge_gap_frames, fps) for t in candidates]
    )


def threshold_for_rate(
    videos: Sequence[Video],
    target: float,
    candidates: FloatArray,
    min_alert_frames: int,
    merge_gap_frames: int,
    fps: float,
) -> tuple[float, bool]:
    """Lowest candidate (ascending) whose rate is at most `target` alerts per hour, and
    True; the largest candidate and False if none reaches it."""
    rates = alert_rate_curve(videos, candidates, min_alert_frames, merge_gap_frames, fps)
    ok = np.flatnonzero(rates <= target)
    if len(ok):
        return float(candidates[ok[0]]), True
    return float(candidates[-1]), False
