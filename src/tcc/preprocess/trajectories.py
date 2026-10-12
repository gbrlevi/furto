"""Per-person trajectories: gap interpolation up to k frames and short-track removal.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from tcc.data.schema import VideoPoses
from tcc.data.skeleton import V


class Trajectory(NamedTuple):
    track_id: int
    frames: npt.NDArray[np.int64]  # (n,) consecutive frame indices
    keypoints: npt.NDArray[np.float32]  # (n, 17, 3)


def tracks(video: VideoPoses) -> list[Trajectory]:
    """Rows of each person, sorted by frame, in order of first appearance."""
    _, first = np.unique(video.track_id, return_index=True)
    out = []
    for tid in video.track_id[np.sort(first)]:
        rows = np.flatnonzero(video.track_id == tid)
        rows = rows[np.argsort(video.frame_idx[rows], kind="stable")]
        out.append(Trajectory(int(tid), video.frame_idx[rows], video.keypoints[rows]))
    return out


def _interpolate(
    frames: npt.NDArray[np.int64], keypoints: npt.NDArray[np.float32], target: npt.NDArray[np.int64]
) -> npt.NDArray[np.float32]:
    res = np.empty((len(target), V * 3), np.float32)
    flat = keypoints.reshape(len(frames), -1)
    for c in range(flat.shape[1]):
        res[:, c] = np.interp(target, frames, flat[:, c])
    return res.reshape(len(target), V, 3)


def split_trajectories(tracks: list[Trajectory], max_gap: int, min_length: int) -> list[Trajectory]:
    """Cut each track where more than `max_gap` frames are missing, drop pieces spanning
    fewer than `min_length` frames and fill the remaining gaps by linear interpolation of
    x, y and confidence.
    """
    out = []
    for tid, frames, keypoints in tracks:
        cuts = np.where(np.diff(frames) > max_gap + 1)[0] + 1
        for qs, ks in zip(np.split(frames, cuts), np.split(keypoints, cuts), strict=True):
            if qs[-1] - qs[0] + 1 < min_length:
                continue
            full = np.arange(qs[0], qs[-1] + 1)
            if len(full) != len(qs):
                ks = _interpolate(qs, ks, full)
            out.append(Trajectory(tid, full, ks))
    return out
