"""Sliding windows over trajectories, normalized and laid out as (N, 3, T, 17).

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from tcc.data.schema import VideoPoses
from tcc.data.skeleton import V
from tcc.preprocess.normalize import ScaleGuard, guarded_scale, normalize, window_scale
from tcc.preprocess.quality import SCORABLE, gate_reasons, window_quality
from tcc.preprocess.trajectories import Trajectory, split_trajectories, tracks


class TheftInTrainingError(RuntimeError):
    pass


def window_starts(n: int, length: int, stride: int) -> list[int]:
    """Starts every `stride` frames; the last window always ends at the last frame."""
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] != n - length:
        starts.append(n - length)
    return starts


class RawWindows(NamedTuple):
    keypoints: npt.NDArray[np.float32]  # (N, T, 17, 3) before normalization
    boxes: npt.NDArray[np.float32]  # (N, T, 4)
    real: npt.NDArray[np.bool_]  # (N, T) real detection, not interpolated
    meta: npt.NDArray[np.int64]  # (N, 3) [video, track, first frame]


def raw_windows(
    trajectories: list[Trajectory], length: int, stride: int, video_idx: int = 0
) -> RawWindows:
    """Windows of the trajectories before normalization."""
    kps, boxes, real, meta = [], [], [], []
    for t in trajectories:
        n = len(t.frames)
        b = np.zeros((n, 4), np.float32) if t.boxes is None else t.boxes
        r = np.ones(n, np.bool_) if t.real is None else t.real
        for s in window_starts(n, length, stride):
            kps.append(t.keypoints[s : s + length])
            boxes.append(b[s : s + length])
            real.append(r[s : s + length])
            meta.append((video_idx, t.track_id, int(t.frames[s])))
    if not kps:
        return RawWindows(
            np.zeros((0, length, V, 3), np.float32),
            np.zeros((0, length, 4), np.float32),
            np.zeros((0, length), np.bool_),
            np.zeros((0, 3), np.int64),
        )
    return RawWindows(np.stack(kps), np.stack(boxes), np.stack(real), np.asarray(meta, np.int64))


def make_windows(
    trajectories: list[Trajectory],
    length: int,
    stride: int,
    conf_min: float,
    center: str,
    video_idx: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Normalized windows (N, 3, T, 17) and metadata (N, 3) = [video, track, first frame]."""
    raw = raw_windows(trajectories, length, stride, video_idx)
    if not len(raw.meta):
        return np.zeros((0, 3, length, V), np.float32), raw.meta
    x = normalize(raw.keypoints, conf_min, center).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x), raw.meta


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


SWITCHES = ("on", "off")


class TreatedWindows(NamedTuple):
    x: npt.NDArray[np.float32]  # (N, 3, T, 17) normalized
    meta: npt.NDArray[np.int64]  # (N, 3) [video, track, first frame]
    reason: npt.NDArray[np.int8]  # (N,) quality-gate reason, SCORABLE when the gate is off
    scale_source: npt.NDArray[np.int8]  # (N,) SCALE_* of tcc.preprocess.normalize


def switch(t: Mapping[str, Any], key: str) -> bool:
    """`key: "on" | "off"` of a treatment config; absent means off (configs before 2D)."""
    value = t.get(key, "off")
    if value not in SWITCHES:
        raise ValueError(f"{key} must be one of {SWITCHES} (quoted in YAML), got {value!r}")
    return bool(value == "on")


def treated_windows(
    video: VideoPoses,
    t: Mapping[str, Any],  # Any: the `treatment` section of the YAML config
    stride: int,
    image_size: tuple[int, int],
    real: npt.NDArray[np.bool_] | None = None,
    video_idx: int = 0,
) -> TreatedWindows:
    """Windows of one video with the treatment of `t`, including the scale guard
    (ADR-0015, `scale_guard`) and the quality gate (ADR-0014, `quality_gate`)."""
    trajectories = split_trajectories(tracks(video, real), t["max_gap"], t["T"])
    raw = raw_windows(trajectories, t["T"], stride, video_idx)
    n = len(raw.meta)
    if n == 0:
        empty = np.zeros(0, np.int8)
        return TreatedWindows(np.zeros((0, 3, t["T"], V), np.float32), raw.meta, empty, empty)
    height = raw.boxes[..., 3] - raw.boxes[..., 1]
    guard = ScaleGuard(**t["scale_ratio"]) if switch(t, "scale_guard") else None
    x = normalize(raw.keypoints, t["conf_min"], t["center"], height, guard)
    if guard is None:
        _, source = window_scale(raw.keypoints, t["conf_min"])
    else:
        _, source = guarded_scale(raw.keypoints, height, t["conf_min"], guard)
    reason = np.full(n, SCORABLE, np.int8)
    if switch(t, "quality_gate"):
        quality = window_quality(
            raw.keypoints, raw.boxes, raw.real, image_size, t["conf_min"], t["border_margin_px"]
        )
        reason = gate_reasons(quality, t["quality_thresholds"])
    return TreatedWindows(np.ascontiguousarray(x.transpose(0, 3, 1, 2)), raw.meta, reason, source)
