"""Per-person trajectories: gap interpolation up to k frames and short-track removal.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from tcc.data.schema import VideoPoses


class Trajectory(NamedTuple):
    track_id: int
    frames: npt.NDArray[np.int64]  # (n,) consecutive frame indices
    keypoints: npt.NDArray[np.float32]  # (n, 17, 3)
    boxes: npt.NDArray[np.float32] | None = None  # (n, 4) x1, y1, x2, y2
    real: npt.NDArray[np.bool_] | None = None  # (n,) a real detection, not interpolated


def tracks(video: VideoPoses, real: npt.NDArray[np.bool_] | None = None) -> list[Trajectory]:
    """Rows of each person, sorted by frame, in order of first appearance.

    `real` marks the rows that are real detections (default: all of them); PoseLift rows
    with NaN confidence were interpolated by the authors (ADR-0011).
    """
    if real is None:
        real = np.ones(len(video), np.bool_)
    _, first = np.unique(video.track_id, return_index=True)
    out = []
    for tid in video.track_id[np.sort(first)]:
        rows = np.flatnonzero(video.track_id == tid)
        rows = rows[np.argsort(video.frame_idx[rows], kind="stable")]
        out.append(
            Trajectory(
                int(tid), video.frame_idx[rows], video.keypoints[rows], video.bbox[rows], real[rows]
            )
        )
    return out


def _interpolate(
    frames: npt.NDArray[np.int64], values: npt.NDArray[np.float32], target: npt.NDArray[np.int64]
) -> npt.NDArray[np.float32]:
    res = np.empty((len(target), values[0].size), np.float32)
    flat = values.reshape(len(frames), -1)
    for c in range(flat.shape[1]):
        res[:, c] = np.interp(target, frames, flat[:, c])
    return res.reshape(len(target), *values.shape[1:])


def split_trajectories(tracks: list[Trajectory], max_gap: int, min_length: int) -> list[Trajectory]:
    """Cut each track where more than `max_gap` frames are missing, drop pieces spanning
    fewer than `min_length` frames and fill the remaining gaps by linear interpolation of
    x, y, confidence and box. Filled frames are marked as not real.
    """
    out = []
    for tid, frames, keypoints, boxes, real in tracks:
        n = len(frames)
        boxes = np.zeros((n, 4), np.float32) if boxes is None else boxes
        real = np.ones(n, np.bool_) if real is None else real
        cuts = np.where(np.diff(frames) > max_gap + 1)[0] + 1
        pieces = zip(
            np.split(frames, cuts),
            np.split(keypoints, cuts),
            np.split(boxes, cuts),
            np.split(real, cuts),
            strict=True,
        )
        for qs, ks, bs, rs in pieces:
            if qs[-1] - qs[0] + 1 < min_length:
                continue
            full = np.arange(qs[0], qs[-1] + 1)
            if len(full) != len(qs):
                ks = _interpolate(qs, ks, full)
                bs = _interpolate(qs, bs, full)
                observed = np.zeros(len(full), np.bool_)
                observed[qs - qs[0]] = rs
                rs = observed
            out.append(Trajectory(tid, full, ks, bs, rs))
    return out
