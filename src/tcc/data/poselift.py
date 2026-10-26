"""Explicit reader for the PoseLift pickles (docs/tcc/FORMATO_POSELIFT.md).

Layout under the dataset root, as distributed on the authors' Google Drive:

    Pickle_files/Train/<camera>_<video>.pkl
    Pickle_files/Test/<camera>_<video>.pkl
    Pickle_files/GT/<camera>_<video>.npy      (test only; one 0/1 label per frame)

Each pickle is {frame_idx: {person_id: [bbox, keypoints]}}, with frame_idx from 0 and
contiguous, bbox as four integers and keypoints as a (17, 3) float16 array. Anything
else raises `PoseLiftFormatError`. The column order of keypoints and bbox comes from
the config, because the published README and the files disagree (ADR-0010). NaN
confidences mark poses interpolated by the authors and are handled by `nan_confidence`
(ADR-0011).

`sanitize` is adapted from luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import dataclasses
import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tcc.data.schema import VideoPoses
from tcc.data.skeleton import V

SPLIT_DIRS = {"train": "Train", "test": "Test"}
LABEL_DIR = "GT"
VIDEO_ID = re.compile(r"^(\d+)_(\d+)$")

# Column positions of (x, y) inside the stored keypoint triplets.
# yx_stored: production (ADR-0010); as_is: columns as stored, reproduces the original PoC.
KEYPOINT_ORDERS = {"yx_stored": (1, 0), "as_is": (0, 1)}
BBOX_FORMATS = ("xyxy", "xywh")
# interpolate: production (ADR-0011); one: original PoC; zero: sensitivity on validation;
# keep: leave NaN untouched, for inspection only.
NAN_CONFIDENCE = ("interpolate", "one", "zero", "keep")

# The pickles only need numpy's array reconstruction; anything else is refused.
_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class PoseLiftFormatError(ValueError):
    pass


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # Any: pickle's own signature
        if (module, name) not in _ALLOWED_GLOBALS:
            raise PoseLiftFormatError(f"pickle requests forbidden global {module}.{name}")
        return super().find_class(module, name)


def read_pickle(path: Path) -> Any:  # Any: raw pickle content, validated by parse_video
    with path.open("rb") as f:
        return _RestrictedUnpickler(f).load()


def pickle_path(root: Path, split: str, video_id: str) -> Path:
    return root / "Pickle_files" / SPLIT_DIRS[split] / f"{video_id}.pkl"


def label_path(root: Path, video_id: str) -> Path:
    return root / "Pickle_files" / LABEL_DIR / f"{video_id}.npy"


def camera_of(video_id: str) -> str:
    match = VIDEO_ID.match(video_id)
    if match is None:
        raise PoseLiftFormatError(f"video id {video_id!r} is not <camera>_<video>")
    return match.group(1)


def sanitize(
    keypoints: npt.NDArray[np.float32], conf_nan: float | None
) -> tuple[npt.NDArray[np.float32], int, int]:
    """Handle undefined values (NaN/inf) in (..., 17, 3) keypoints.

    - x or y undefined: missing point (coordinates and confidence set to 0);
    - only the confidence undefined, x and y valid: confidence = `conf_nan`.
    With `conf_nan=None` the array is returned unchanged (for inspection).
    Returns (keypoints, number of confidence NaN, number of x/y NaN).
    """
    kp = keypoints.astype(np.float32, copy=True)
    xy_bad = ~np.isfinite(kp[..., :2]).all(-1)
    conf_bad = ~np.isfinite(kp[..., 2]) & ~xy_bad
    if conf_nan is not None:
        kp[conf_bad, 2] = conf_nan
        kp[xy_bad] = 0.0
    return kp, int(conf_bad.sum()), int(xy_bad.sum())


def interpolate_nan_confidence(
    video_id: str,
    frame_idx: npt.NDArray[np.int64],
    track_id: npt.NDArray[np.int64],
    keypoints: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Replace NaN confidences by linear interpolation between the real poses of the same
    track, joint by joint (ADR-0011). A NaN without a real pose on both sides is an error.
    """
    out = keypoints.copy()
    nan = np.isnan(out[..., 2])
    for tid in np.unique(track_id[nan.any(1)]):
        rows = np.flatnonzero(track_id == tid)
        rows = rows[np.argsort(frame_idx[rows])]
        q = frame_idx[rows]
        for j in range(out.shape[1]):
            c = out[rows, j, 2]
            bad = np.isnan(c)
            if not bad.any():
                continue
            good = ~bad
            if not good.any() or q[bad].min() < q[good].min() or q[bad].max() > q[good].max():
                raise PoseLiftFormatError(
                    f"{video_id}: track {tid} joint {j} has NaN confidence without real "
                    "poses on both sides"
                )
            out[rows[bad], j, 2] = np.interp(q[bad], q[good], c[good])
    return out


def resolve_nan_confidence(
    video_id: str,
    frame_idx: npt.NDArray[np.int64],
    track_id: npt.NDArray[np.int64],
    keypoints: npt.NDArray[np.float32],
    nan_confidence: str,
) -> npt.NDArray[np.float32]:
    """Apply the `nan_confidence` policy to (N, 17, 3) keypoints (ADR-0011)."""
    if nan_confidence not in NAN_CONFIDENCE:
        raise ValueError(f"nan_confidence must be one of {NAN_CONFIDENCE}")
    if nan_confidence == "interpolate":
        keypoints = interpolate_nan_confidence(video_id, frame_idx, track_id, keypoints)
    if nan_confidence != "keep":
        keypoints, _, _ = sanitize(keypoints, 0.0 if nan_confidence == "zero" else 1.0)
    return keypoints


def parse_video(
    video_id: str,
    raw: Any,  # Any: unvalidated pickle content
    labels: npt.NDArray[np.int64] | None,
    keypoint_order: str,
    bbox_format: str,
    nan_confidence: str,
) -> VideoPoses:
    if keypoint_order not in KEYPOINT_ORDERS:
        raise ValueError(f"keypoint_order must be one of {sorted(KEYPOINT_ORDERS)}")
    if bbox_format not in BBOX_FORMATS:
        raise ValueError(f"bbox_format must be one of {BBOX_FORMATS}")
    if nan_confidence not in NAN_CONFIDENCE:
        raise ValueError(f"nan_confidence must be one of {NAN_CONFIDENCE}")
    if not isinstance(raw, dict) or not raw:
        raise PoseLiftFormatError(f"{video_id}: top level must be a non-empty dict of frames")
    frames = list(raw)
    if not all(type(q) is int for q in frames):
        raise PoseLiftFormatError(f"{video_id}: frame keys must be int")
    if frames != list(range(len(frames))):
        raise PoseLiftFormatError(f"{video_id}: frame keys must be 0..n-1 in order")

    rows_q, rows_pid, rows_box, rows_kp = [], [], [], []
    for q, persons in raw.items():
        if not isinstance(persons, dict):
            raise PoseLiftFormatError(f"{video_id}: frame {q} must map person ids to entries")
        for pid, entry in persons.items():
            if type(pid) is not int:
                raise PoseLiftFormatError(f"{video_id}: frame {q} has non-int person id {pid!r}")
            if not isinstance(entry, list | tuple) or len(entry) != 2:
                raise PoseLiftFormatError(f"{video_id}: frame {q} person {pid} is not [bbox, kp]")
            box = np.asarray(entry[0], dtype=np.float64)
            kp = entry[1]
            if box.shape != (4,):
                raise PoseLiftFormatError(
                    f"{video_id}: frame {q} person {pid} bbox shape {box.shape}"
                )
            if not isinstance(kp, np.ndarray) or kp.shape != (V, 3):
                shape = getattr(kp, "shape", type(kp).__name__)
                raise PoseLiftFormatError(f"{video_id}: frame {q} person {pid} keypoints {shape}")
            rows_q.append(q)
            rows_pid.append(pid)
            rows_box.append(box)
            rows_kp.append(kp.astype(np.float32))

    n_frames = len(frames)
    if labels is not None:
        if labels.ndim != 1 or len(labels) < n_frames:
            raise PoseLiftFormatError(
                f"{video_id}: {len(labels)} labels for {n_frames} pose frames"
            )
        n_frames = len(labels)

    keypoints = np.stack(rows_kp) if rows_kp else np.zeros((0, V, 3), np.float32)
    ix, iy = KEYPOINT_ORDERS[keypoint_order]
    keypoints = keypoints[..., [ix, iy, 2]]
    frame_idx = np.asarray(rows_q, np.int64)
    track_id = np.asarray(rows_pid, np.int64)
    keypoints = resolve_nan_confidence(video_id, frame_idx, track_id, keypoints, nan_confidence)
    boxes = np.stack(rows_box) if rows_box else np.zeros((0, 4))
    if bbox_format == "xywh":
        boxes = np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], axis=1)

    return VideoPoses(
        video_id=video_id,
        camera=camera_of(video_id),
        n_frames=n_frames,
        frame_idx=frame_idx,
        track_id=track_id,
        bbox=boxes.astype(np.float32),
        keypoints=np.ascontiguousarray(keypoints, dtype=np.float32),
        frame_labels=labels,
    )


def load_labels(root: Path, video_id: str) -> npt.NDArray[np.int64]:
    labels: npt.NDArray[np.int64] = np.load(label_path(root, video_id), allow_pickle=False)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise PoseLiftFormatError(f"{video_id}: labels must be a 1-D 0/1 array")
    return labels.astype(np.int64)


def load_video(root: Path, split: str, video_id: str, cfg: Mapping[str, Any]) -> VideoPoses:
    """Load one video. `cfg` holds keypoint_order, bbox_format and nan_confidence."""
    labels = load_labels(root, video_id) if split == "test" else None
    return parse_video(
        video_id,
        read_pickle(pickle_path(root, split, video_id)),
        labels,
        keypoint_order=cfg["keypoint_order"],
        bbox_format=cfg["bbox_format"],
        nan_confidence=cfg["nan_confidence"],
    )


def load_video_with_real(
    root: Path, split: str, video_id: str, cfg: Mapping[str, Any]
) -> tuple[VideoPoses, npt.NDArray[np.bool_]]:
    """`load_video` plus a (N,) mask of real detections: rows whose confidence the authors
    left as NaN are poses they interpolated (ADR-0011) and are not real."""
    kept = load_video(root, split, video_id, {**cfg, "nan_confidence": "keep"})
    real = np.asarray(~np.isnan(kept.keypoints[..., 2]).any(1), np.bool_)
    keypoints = resolve_nan_confidence(
        video_id, kept.frame_idx, kept.track_id, kept.keypoints, cfg["nan_confidence"]
    )
    return dataclasses.replace(kept, keypoints=np.ascontiguousarray(keypoints)), real
