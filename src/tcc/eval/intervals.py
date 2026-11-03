"""Event-level matching of predicted and true frame intervals (inclusive ends)."""

from collections.abc import Sequence

Span = tuple[int, int]


def iou(a: Span, b: Span) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    return inter / (max(a[1], b[1]) - min(a[0], b[0]) + 1)


def match(pred: Sequence[Span], truth: Sequence[Span], min_iou: float) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) with greedy one-to-one matching,
    best overlap first; a pair counts only with IoU >= `min_iou`."""
    pairs = sorted(
        ((iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth)),
        reverse=True,
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    for value, i, j in pairs:
        if value < min_iou or value == 0:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    tp = len(used_p)
    return tp, len(pred) - tp, len(truth) - tp
