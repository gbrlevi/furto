"""Frame metrics: AUC-ROC, AUC-PR, EER and metrics at a threshold.

Human-review zone (CLAUDE.md §6). Scores are anomaly scores (higher = more anomalous)
and labels are 1 for theft. The AUC-PR has two declared versions (ADR-0008): theft as the
positive class (main) and normal as the positive class (comparison with Table 3 of the
PoseLift paper, as in the distributed STG-NF code).

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

POSITIVE_CLASSES = ("theft", "normal")
PR_ESTIMATORS = ("trapezoid", "average_precision")


def auc_roc(y: IntArray, s: FloatArray) -> float:
    return float(roc_auc_score(y, s))


def auc_pr(y: IntArray, s: FloatArray, positive: str, estimator: str) -> float:
    """Area under the precision-recall curve. `positive: normal` uses the normal frames as
    positives and the negated score. `trapezoid` is sklearn's `auc(recall, precision)`, as
    in the STG-NF code; `average_precision` is the original PoC estimator."""
    if positive not in POSITIVE_CLASSES:
        raise ValueError(f"positive must be one of {POSITIVE_CLASSES}, got {positive!r}")
    if estimator not in PR_ESTIMATORS:
        raise ValueError(f"estimator must be one of {PR_ESTIMATORS}, got {estimator!r}")
    if positive == "normal":
        y, s = 1 - y, -s
    if estimator == "average_precision":
        return float(average_precision_score(y, s))
    precision, recall, _ = precision_recall_curve(y, s)
    return float(auc(recall, precision))


def eer(y: IntArray, s: FloatArray) -> float:
    """Equal error rate: mean of the false positive and false negative rates at the ROC
    point where they are closest."""
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def threshold_metrics(y: IntArray, s: FloatArray, threshold: float) -> dict[str, Any]:
    """Precision, recall, specificity, F1 and H-PRS (harmonic mean of precision, recall and
    specificity; RetailS, Sec. III-B2) of `score > threshold`."""
    p = s > threshold
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    parts = (precision, recall, specificity)
    h_prs = 3 / sum(1 / v for v in parts) if min(parts) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "h_prs": h_prs,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
