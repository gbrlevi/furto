"""Confidence interval of the frame AUC-ROC by bootstrap over videos (session 3, B4).

Human-review zone (CLAUDE.md §6).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from tcc.eval.metrics import FloatArray, IntArray, auc_roc


def bootstrap_auc_by_video(
    labels: Sequence[IntArray],
    scores: Sequence[FloatArray],
    n_resamples: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    """Percentile interval of the AUC-ROC resampling whole videos with replacement.

    Frames of one video are strongly correlated; resampling frames would narrow the
    interval artificially. Resamples with a single class have no AUC and are counted.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs, one_class = [], 0
    for _ in range(n_resamples):
        pick = rng.integers(0, n, n)
        y = np.concatenate([labels[i] for i in pick])
        if y.min() == y.max():
            one_class += 1
            continue
        aucs.append(auc_roc(y, np.concatenate([scores[i] for i in pick])))
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(aucs, [tail, 100 - tail]) if aucs else (np.nan, np.nan)
    return {
        "low": float(low),
        "high": float(high),
        "level": level,
        "resamples_used": len(aucs),
        "resamples_one_class": one_class,
        "seed": seed,
    }
