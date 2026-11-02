"""Post-hoc checks of a frame score: no-model baselines and the AUC inside each video.

The pooled frame AUC mixes videos, so it also rewards a score that only tells videos (or
scenes) apart. Here the score is compared with what a count of people gives, and with
itself inside each video, against a null that rolls the labels within each video.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def auc(labels: IntArray, scores: FloatArray) -> float:
    """AUC-ROC by the rank-sum formula (ties count half); NaN if only one class."""
    positive = labels == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def people_per_frame(frame_idx: IntArray, n_frames: int) -> IntArray:
    """Number of pose rows in each frame of a video."""
    count = np.zeros(n_frames, np.int64)
    np.add.at(count, frame_idx, 1)
    return count


def within_video_auc(
    labels: Sequence[IntArray], scores: Sequence[FloatArray], masks: Sequence[BoolArray]
) -> tuple[float, int]:
    """Mean over videos of the AUC on the masked frames; videos with one class are left out."""
    values = [auc(y[m], s[m]) for y, s, m in zip(labels, scores, masks, strict=True)]
    kept = [v for v in values if np.isfinite(v)]
    return (float(np.mean(kept)) if kept else float("nan")), len(kept)


def within_video_null(
    labels: Sequence[IntArray],
    scores: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    draws: int,
    seed: int,
) -> FloatArray:
    """`within_video_auc` with each video's labels rolled by a random non-zero shift: keeps
    the length and position structure of the theft intervals, breaks their timing."""
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for d in range(draws):
        rolled = [np.roll(y, int(rng.integers(1, len(y)))) for y in labels]
        out[d] = within_video_auc(rolled, scores, masks)[0]
    return out


def events_with_room(labels: IntArray, scored: BoolArray, min_run: int, after: int) -> int:
    """Theft intervals with at least `min_run` consecutive scored frames between their start
    and `after` frames past their end: the only events an alert of `min_run` frames can hit."""
    edges = np.diff(np.concatenate([[0], labels, [0]]))
    count = 0
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        run = best = 0
        for f in range(start, min(end + after, len(scored))):
            run = run + 1 if scored[f] else 0
            best = max(best, run)
        count += best >= min_run
    return count
