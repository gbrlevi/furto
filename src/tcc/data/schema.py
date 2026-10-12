"""Pose data contract shared by the PoseLift loader and the video pipeline (ADR-0004).

One row per person per frame. Rows of one video are stored column-wise.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tcc.data.skeleton import V


@dataclass(frozen=True)
class PoseRecord:
    video_id: str
    frame_idx: int  # at 15 fps, base 0
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    keypoints: npt.NDArray[np.float32]  # (17, 3): x, y in pixels, confidence
    label: int | None  # 0 normal, 1 theft, None unlabelled


@dataclass
class VideoPoses:
    """All records of one video, column-wise. Row i is one `PoseRecord`."""

    video_id: str
    camera: str
    n_frames: int
    frame_idx: npt.NDArray[np.int64]  # (N,)
    track_id: npt.NDArray[np.int64]  # (N,)
    bbox: npt.NDArray[np.float32]  # (N, 4) x1, y1, x2, y2
    keypoints: npt.NDArray[np.float32]  # (N, 17, 3) x, y, confidence
    frame_labels: npt.NDArray[np.int64] | None  # (n_frames,) 0/1, or None

    def __post_init__(self) -> None:
        validate(self)

    def __len__(self) -> int:
        return len(self.frame_idx)

    def records(self) -> Iterator[PoseRecord]:
        for i in range(len(self)):
            q = int(self.frame_idx[i])
            x1, y1, x2, y2 = (float(v) for v in self.bbox[i])
            yield PoseRecord(
                video_id=self.video_id,
                frame_idx=q,
                track_id=int(self.track_id[i]),
                bbox=(x1, y1, x2, y2),
                keypoints=self.keypoints[i],
                label=None if self.frame_labels is None else int(self.frame_labels[q]),
            )


def validate(v: VideoPoses) -> None:
    """Raise ValueError if `v` breaks the contract."""
    n = len(v.frame_idx)
    checks = [
        (v.frame_idx.shape == (n,) and v.frame_idx.dtype == np.int64, "frame_idx must be int64 (N,)"),
        (v.track_id.shape == (n,) and v.track_id.dtype == np.int64, "track_id must be int64 (N,)"),
        (v.bbox.shape == (n, 4) and v.bbox.dtype == np.float32, "bbox must be float32 (N, 4)"),
        (
            v.keypoints.shape == (n, V, 3) and v.keypoints.dtype == np.float32,
            "keypoints must be float32 (N, 17, 3)",
        ),
        (n == 0 or (v.frame_idx.min() >= 0 and v.frame_idx.max() < v.n_frames), "frame_idx out of range"),
        (bool(np.all(v.bbox[:, 2] >= v.bbox[:, 0])), "bbox x2 < x1"),
        (bool(np.all(v.bbox[:, 3] >= v.bbox[:, 1])), "bbox y2 < y1"),
    ]
    if v.frame_labels is not None:
        labels = v.frame_labels
        checks += [
            (labels.shape == (v.n_frames,), "frame_labels must have one value per frame"),
            (bool(np.isin(labels, (0, 1)).all()), "frame_labels must be 0 or 1"),
        ]
    pairs = np.stack([v.frame_idx, v.track_id], axis=1)
    checks.append((len(np.unique(pairs, axis=0)) == n, "duplicate (frame_idx, track_id) row"))
    for ok, message in checks:
        if not ok:
            raise ValueError(f"{v.video_id}: {message}")
