"""Window scores to frame scores, the most normal fill and smoothing (ROADMAP stage 4).

Human-review zone (CLAUDE.md §6). Offline evaluation follows the STG-NF protocol
distributed with PoseLift (ADR-0008): the window score goes to its centre frame, the frame
score is the maximum over people, frames without a score (nobody, or only windows filtered
by the quality gate, ADR-0014) get the most normal score of the evaluated set, then the
gaussian filter is applied once per sigma, in sequence. `last` is the causal operation
mode; `max_cover` is the original PoC rule, kept for the port equivalence.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

FloatArray = npt.NDArray[np.float64]

FRAME_SCORE_MODES = ("center", "last", "max_cover")


def window_frame_scores(
    meta: npt.NDArray[np.int64],
    scores: npt.NDArray[np.floating],
    length: int,
    n_frames: int,
    mode: str,
) -> FloatArray:
    """(n_frames,) maximum over people and windows of the window scores attributed to each
    frame; NaN where no finite score lands. meta rows are [video, track, first frame];
    a NaN score (a not-scorable window) is ignored."""
    if mode not in FRAME_SCORE_MODES:
        raise ValueError(f"mode must be one of {FRAME_SCORE_MODES}, got {mode!r}")
    out = np.full(n_frames, np.nan)
    ok = np.isfinite(scores)
    starts, values = meta[ok, 2], np.asarray(scores, np.float64)[ok]
    if mode == "max_cover":
        for start, s in zip(starts.tolist(), values.tolist(), strict=True):
            end = min(start + length, n_frames)
            out[start:end] = np.fmax(out[start:end], s)
        return out
    frames = starts + (length // 2 if mode == "center" else length - 1)
    inside = (frames >= 0) & (frames < n_frames)
    best = np.full(n_frames, -np.inf)
    np.maximum.at(best, frames[inside], values[inside])
    hit = np.isfinite(best)
    out[hit] = best[hit]
    return out


def most_normal(frame_scores: Sequence[FloatArray]) -> float:
    """Smallest finite score over all videos of the evaluated set (STG-NF replaces the
    frames without a score by the most normal score of the whole set)."""
    finite = [s[np.isfinite(s)] for s in frame_scores]
    values = np.concatenate(finite) if finite else np.zeros(0)
    if len(values) == 0:
        raise ValueError("no finite frame score")
    return float(values.min())


def fill(frame_scores: FloatArray, value: float) -> FloatArray:
    """Frames without a score (NaN) get `value`."""
    return np.where(np.isfinite(frame_scores), frame_scores, value)


def smooth(frame_scores: FloatArray, sigmas: Sequence[float]) -> FloatArray:
    """Gaussian filter applied once per sigma, in order (STG-NF: sigma = 1, ..., 6)."""
    out = np.asarray(frame_scores, np.float64)
    for sigma in sigmas:
        out = gaussian_filter1d(out, sigma)
    return out


def post_process(
    raw: Sequence[FloatArray], fill_value: float, sigmas: Sequence[float]
) -> list[FloatArray]:
    """Fill and smooth each video (the original `pos_processar` without camera norms)."""
    return [smooth(fill(s, fill_value), sigmas) for s in raw]
