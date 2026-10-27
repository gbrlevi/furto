"""Event metrics: theft events detected by an alert and alerts with no event.

Human-review zone (CLAUDE.md §6). Not in the Relatório (proposed in the port).

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

Interval = tuple[int, int]  # [start, end)


def label_intervals(labels: npt.NDArray[np.int64]) -> list[Interval]:
    """[start, end) runs of theft frames."""
    edges = np.diff(np.concatenate([[0], (labels == 1).astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _near(a: Interval, b: Interval, tolerance: int) -> bool:
    return a[0] < b[1] + tolerance and a[1] > b[0] - tolerance


def event_metrics(
    labels: Sequence[npt.NDArray[np.int64]],
    alert_intervals: Sequence[Sequence[Interval]],
    tolerance: int,
) -> dict[str, Any]:
    """An event is detected if some alert overlaps it, allowing `tolerance` frames on each
    side; an alert is false if it overlaps no event in the same way."""
    events = detected = n_alerts = false = 0
    for y, found in zip(labels, alert_intervals, strict=True):
        theft = label_intervals(y)
        events += len(theft)
        n_alerts += len(found)
        detected += sum(any(_near(a, e, tolerance) for a in found) for e in theft)
        false += sum(not any(_near(a, e, tolerance) for e in theft) for a in found)
    return {
        "events": events,
        "events_detected": detected,
        "event_recall": detected / events if events else float("nan"),
        "alerts": n_alerts,
        "false_alerts": false,
    }
