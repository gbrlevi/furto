"""Alert logic on causal frame scores (session 2D, task 6).

Human-review zone (CLAUDE.md §6). An alert needs `min_alert_frames` consecutive scored
frames above the threshold; frames that are not scored (nobody, or only windows filtered
by the quality gate, ADR-0014) break the run. Alerts closer than `merge_gap_frames` are
merged. The frame AUC is not affected; only the operating point is.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class Alert(NamedTuple):
    start: int  # first frame of the run above the threshold
    end: int  # one past the last frame above the threshold
    fired: int  # frame where the alert is raised: start + min_alert_frames - 1


def runs_above(
    score: npt.NDArray[np.float64], scored: npt.NDArray[np.bool_], threshold: float
) -> list[tuple[int, int]]:
    """[start, end) runs of consecutive scored frames with score above the threshold."""
    above = scored & (np.nan_to_num(score, nan=-np.inf) > threshold)
    edges = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def alerts(
    score: npt.NDArray[np.float64],
    scored: npt.NDArray[np.bool_],
    threshold: float,
    min_alert_frames: int,
    merge_gap_frames: int,
) -> list[Alert]:
    """Runs of at least `min_alert_frames`, merged when the gap between one alert's end and
    the next one's start is shorter than `merge_gap_frames`."""
    out: list[Alert] = []
    for start, end in runs_above(score, scored, threshold):
        if end - start < min_alert_frames:
            continue
        if out and start - out[-1].end < merge_gap_frames:
            out[-1] = out[-1]._replace(end=end)
        else:
            out.append(Alert(start, end, start + min_alert_frames - 1))
    return out
