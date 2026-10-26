"""Operational scoring of extracted poses (session 2C demonstration).

Causal mode of ADR-0008: the window score goes to the window's last frame; the frame
score is the maximum over the people present. No smoothing (not defined for operation
yet). Session 2D adds the "not scorable" state of the quality gate (ADR-0014): a frame
whose windows were all filtered gets the most normal score and says why.
Human-review zone (CLAUDE.md §6): written for the 2C and 2D demos, to be reviewed in
session 3 together with the evaluation module.
"""

from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from tcc.data.schema import VideoPoses

FloatArray = npt.NDArray[np.float64]

# Frame states. NO_WINDOW: nobody, or nobody with a complete window ending here.
NO_WINDOW, SCORED, NOT_SCORABLE = 0, 1, 2
STATES = {NO_WINDOW: "no_window", SCORED: "scored", NOT_SCORABLE: "not_scorable"}
MOST_NORMAL = -np.inf  # score of frames without a scorable window (ADR-0008, ADR-0014)


def load_extracted(path: Path, video_id: str, camera: str) -> VideoPoses:
    """Read a `<id>_poses.npz` written by scripts/extract_pose.py (ADR-0004 contract)."""
    z = np.load(path)
    return VideoPoses(
        video_id=video_id,
        camera=camera,
        n_frames=int(z["n_frames"]),
        frame_idx=z["frame_idx"].astype(np.int64),
        track_id=z["track_id"].astype(np.int64),
        bbox=z["bbox"].astype(np.float32),
        keypoints=z["keypoints"].astype(np.float32),
        frame_labels=None,
    )


def last_frame_scores(
    meta: npt.NDArray[np.int64], scores: npt.NDArray[np.float32], length: int
) -> dict[tuple[int, int], float]:
    """{(track_id, frame): score} with each window's score on its last frame.

    meta rows are [video, track_id, first frame], as in `make_windows`.
    """
    out: dict[tuple[int, int], float] = {}
    for (_, tid, start), s in zip(meta.tolist(), scores.tolist(), strict=True):
        if np.isfinite(s):
            key = (int(tid), int(start) + length - 1)
            out[key] = max(out.get(key, -np.inf), float(s))
    return out


def last_frame_reasons(
    meta: npt.NDArray[np.int64], reason: npt.NDArray[np.int8], length: int
) -> dict[tuple[int, int], int]:
    """{(track_id, frame): reason} of the not-scorable windows (reason != 0), each on its
    last frame."""
    out: dict[tuple[int, int], int] = {}
    for (_, tid, start), r in zip(meta.tolist(), reason.tolist(), strict=True):
        if r:
            out[(int(tid), int(start) + length - 1)] = int(r)
    return out


class FrameScores(NamedTuple):
    score: FloatArray  # (n_frames,) maximum over scored people; MOST_NORMAL otherwise
    state: npt.NDArray[np.int8]  # (n_frames,) NO_WINDOW, SCORED or NOT_SCORABLE
    reason: npt.NDArray[np.int8]  # (n_frames,) most frequent reason of NOT_SCORABLE frames


def frame_scores(
    per_person: dict[tuple[int, int], float],
    per_person_reason: dict[tuple[int, int], int],
    n_frames: int,
) -> FrameScores:
    """Frame score, state and reason. A frame where nobody has a scorable window but
    someone has a filtered one is NOT_SCORABLE and gets MOST_NORMAL, like an empty frame."""
    fm = frame_max(per_person, n_frames)
    scored = np.isfinite(fm)
    reasons: dict[int, Counter[int]] = {}
    for (_, q), r in per_person_reason.items():
        if 0 <= q < n_frames and not scored[q]:
            reasons.setdefault(q, Counter())[r] += 1
    state = np.where(scored, SCORED, NO_WINDOW).astype(np.int8)
    reason = np.zeros(n_frames, np.int8)
    for q, counts in reasons.items():
        state[q] = NOT_SCORABLE
        reason[q] = min(counts, key=lambda r: (-counts[r], r))
    return FrameScores(np.where(scored, fm, MOST_NORMAL), state, reason)


def frame_max(per_person: dict[tuple[int, int], float], n_frames: int) -> FloatArray:
    """Frame score = maximum over people; NaN where nobody has a score."""
    out = np.full(n_frames, np.nan)
    for (_, q), s in per_person.items():
        if 0 <= q < n_frames:
            out[q] = s if np.isnan(out[q]) else max(out[q], s)
    return out


def quantile_threshold(scores: npt.NDArray[np.float32], q: float) -> float:
    """Provisional threshold: quantile `q` of finite (validation) window scores."""
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        raise ValueError("no finite score to take a quantile from")
    return float(np.quantile(finite, q))


def alert_intervals(frame_scores: FloatArray, threshold: float) -> list[tuple[int, int]]:
    """[start, end) runs of consecutive frames with score above the threshold."""
    above = np.nan_to_num(frame_scores, nan=-np.inf) > threshold
    edges = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Half-open intervals [start, end) overlap."""
    return a[0] < b[1] and b[0] < a[1]


def source_interval_to_output(
    keep: npt.NDArray[np.int64], first: int, last: int
) -> tuple[int, int] | None:
    """Output frames [start, end) whose source frame lies in [first, last] (inclusive)."""
    k = np.flatnonzero((keep >= first) & (keep <= last))
    return (int(k[0]), int(k[-1]) + 1) if len(k) else None
