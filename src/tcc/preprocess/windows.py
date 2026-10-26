"""Sliding windows over trajectories, normalized and laid out as (N, 3, T, 17).

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import numpy as np
import numpy.typing as npt

from tcc.data.schema import VideoPoses
from tcc.data.skeleton import V
from tcc.preprocess.normalize import normalize
from tcc.preprocess.trajectories import Trajectory, split_trajectories, tracks


class TheftInTrainingError(RuntimeError):
    pass


def window_starts(n: int, length: int, stride: int) -> list[int]:
    """Starts every `stride` frames; the last window always ends at the last frame."""
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] != n - length:
        starts.append(n - length)
    return starts


def raw_windows(
    trajectories: list[Trajectory], length: int, stride: int, video_idx: int = 0
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Windows before normalization (N, T, 17, 3) and metadata (N, 3) =
    [video, track, first frame]."""
    blocks, meta = [], []
    for tid, frames, keypoints in trajectories:
        for s in window_starts(len(frames), length, stride):
            blocks.append(keypoints[s : s + length])
            meta.append((video_idx, tid, int(frames[s])))
    if not blocks:
        return np.zeros((0, length, V, 3), np.float32), np.zeros((0, 3), np.int64)
    return np.stack(blocks), np.asarray(meta, np.int64)


def make_windows(
    trajectories: list[Trajectory],
    length: int,
    stride: int,
    conf_min: float,
    center: str,
    video_idx: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Normalized windows (N, 3, T, 17) and metadata (N, 3) = [video, track, first frame]."""
    raw, meta = raw_windows(trajectories, length, stride, video_idx)
    if not len(raw):
        return np.zeros((0, 3, length, V), np.float32), meta
    x = normalize(raw, conf_min, center).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x), meta


def check_no_theft(
    meta: npt.NDArray[np.int64], frame_labels: npt.NDArray[np.int64] | None, length: int
) -> None:
    """Raise if any training window touches a frame labelled as theft (porting note 1)."""
    if frame_labels is None or not frame_labels.any() or len(meta) == 0:
        return
    cum = np.concatenate([[0], np.cumsum(frame_labels)])
    end = np.minimum(meta[:, 2] + length, len(frame_labels))
    touched = int(((cum[end] - cum[meta[:, 2]]) > 0).sum())
    if touched:
        raise TheftInTrainingError(f"{touched} training windows touch theft frames")


def video_windows(
    video: VideoPoses,
    length: int,
    stride: int,
    max_gap: int,
    conf_min: float,
    center: str,
    video_idx: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Trajectories of one video cut into normalized windows (the original `gerar_janelas`
    applied to `trajetorias`)."""
    trajectories = split_trajectories(tracks(video), max_gap, length)
    return make_windows(trajectories, length, stride, conf_min, center, video_idx)
