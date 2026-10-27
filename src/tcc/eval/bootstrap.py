"""Confidence interval of a frame metric by bootstrap over videos (session 3, B4).

Human-review zone (CLAUDE.md §6). Rule of P17 item 17: videos with theft and videos
with only normal frames are resampled separately, with replacement and keeping their
counts; in each resample the metric of every seed is computed on the same videos and
averaged; the interval is the percentile interval of that mean.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from tcc.eval.metrics import FloatArray, IntArray


def bootstrap_by_video(
    labels: Sequence[IntArray],
    scores_per_seed: Sequence[Sequence[FloatArray]],
    metric: Callable[[IntArray, FloatArray], float],
    n_resamples: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    """`scores_per_seed[k][v]` is the frame score of video v by seed k. Frames of one video
    are strongly correlated; resampling frames would narrow the interval artificially."""
    theft = [i for i, y in enumerate(labels) if (y == 1).any()]
    normal = [i for i, y in enumerate(labels) if not (y == 1).any()]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_resamples):
        pick = [s[i] for s in (theft, normal) if s for i in rng.integers(0, len(s), len(s))]
        y = np.concatenate([labels[i] for i in pick])
        values = [
            metric(y, np.concatenate([scores[i] for i in pick])) for scores in scores_per_seed
        ]
        means.append(float(np.mean(values)))
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return {
        "low": float(low),
        "high": float(high),
        "level": level,
        "resamples": n_resamples,
        "seed": seed,
        "seeds": len(scores_per_seed),
        "strata": {"theft_videos": len(theft), "normal_videos": len(normal)},
    }
