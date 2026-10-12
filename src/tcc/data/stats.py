"""Descriptive measurements of pose datasets (counts, trajectories, NaN, axis order).

`frame_counts` is adapted from luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from tcc.data.schema import VideoPoses

CONVENTIONS = ("all_frames", "frames_with_person", "person_frames")

# COCO-17 indices used by the axis-order evidence
NOSE, L_SHOULDER, R_SHOULDER, L_ANKLE, R_ANKLE = 0, 5, 6, 15, 16


def frame_counts(videos: Sequence[VideoPoses]) -> dict[str, dict[str, int]]:
    """Normal and theft frames under three counting conventions.

    - all_frames: every frame of the video;
    - frames_with_person: frames with at least one pose;
    - person_frames: one count per person per frame.
    Unlabelled videos count as normal.
    """
    counts = {c: {"normal": 0, "theft": 0} for c in CONVENTIONS}
    for v in videos:
        y = v.frame_labels if v.frame_labels is not None else np.zeros(v.n_frames, np.int64)
        presence = np.bincount(v.frame_idx, minlength=v.n_frames)
        weights = {
            "all_frames": np.ones(v.n_frames, np.int64),
            "frames_with_person": (presence > 0).astype(np.int64),
            "person_frames": presence,
        }
        for name, w in weights.items():
            counts[name]["normal"] += int((w * (y == 0)).sum())
            counts[name]["theft"] += int((w * (y == 1)).sum())
    return counts


def _tracks(v: VideoPoses) -> list[npt.NDArray[np.int64]]:
    """Row indices of each track, sorted by frame."""
    order = np.lexsort((v.frame_idx, v.track_id))
    ids = v.track_id[order]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    return [part for part in np.split(order, cuts) if len(part)]


def track_lengths(videos: Sequence[VideoPoses]) -> dict[str, npt.NDArray[np.int64]]:
    """Per (video, track): frames observed and span; plus lengths of gap-free runs."""
    observed: list[int] = []
    span: list[int] = []
    runs: list[int] = []
    gaps: list[int] = []
    for v in videos:
        for rows in _tracks(v):
            q = v.frame_idx[rows]
            observed.append(len(q))
            span.append(int(q[-1] - q[0] + 1))
            steps = np.diff(q)
            gaps.extend((steps[steps > 1] - 1).tolist())
            cuts = np.flatnonzero(steps > 1) + 1
            runs.extend(len(r) for r in np.split(q, cuts))
    return {
        "observed": np.asarray(observed, np.int64),
        "span": np.asarray(span, np.int64),
        "runs": np.asarray(runs, np.int64),
        "gaps": np.asarray(gaps, np.int64),
    }


def nan_summary(keypoints: npt.NDArray[np.float32]) -> dict[str, Any]:
    """NaN fractions over raw (N, 17, 3) keypoints, before any sanitizing."""
    conf_nan = np.isnan(keypoints[..., 2])
    xy_nan = np.isnan(keypoints[..., 0]) | np.isnan(keypoints[..., 1])
    per_row = conf_nan.sum(1)
    return {
        "points": int(conf_nan.size),
        "conf_nan": int(conf_nan.sum()),
        "conf_nan_fraction": float(conf_nan.mean()) if conf_nan.size else 0.0,
        "xy_nan": int(xy_nan.sum()),
        "xy_nan_fraction": float(xy_nan.mean()) if xy_nan.size else 0.0,
        "rows": int(len(per_row)),
        "rows_by_conf_nan_count": {
            int(k): int(c) for k, c in zip(*np.unique(per_row, return_counts=True), strict=True)
        },
    }


def nan_runs(v: VideoPoses, tolerance_px: float) -> list[dict[str, Any]]:
    """Runs of consecutive rows of one track whose 17 confidences are all NaN.

    For each run: its length, whether it is interior (an observed row of the same
    track right before and right after, in adjacent frames) and, if interior, the
    largest distance in pixels between its x/y and the linear interpolation of the
    two bounding rows. `linear` is that distance <= tolerance_px.
    """
    out = []
    all_nan = np.isnan(v.keypoints[..., 2]).sum(1) == v.keypoints.shape[1]
    for rows in _tracks(v):
        q = v.frame_idx[rows]
        flag = all_nan[rows]
        i = 0
        while i < len(rows):
            if not flag[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(rows) and flag[j + 1] and q[j + 1] == q[j] + 1:
                j += 1
            before, after = i - 1, j + 1
            interior = (
                before >= 0
                and after < len(rows)
                and not flag[before]
                and not flag[after]
                and q[i] == q[before] + 1
                and q[after] == q[j] + 1
            )
            run: dict[str, Any] = {"length": j - i + 1, "interior": bool(interior)}
            if interior:
                a = v.keypoints[rows[before], :, :2].astype(np.float64)
                b = v.keypoints[rows[after], :, :2].astype(np.float64)
                t = (q[i : j + 1] - q[before]) / (q[after] - q[before])
                expected = a[None] + t[:, None, None] * (b - a)[None]
                actual = v.keypoints[rows[i : j + 1], :, :2].astype(np.float64)
                deviation = float(np.nanmax(np.abs(actual - expected)))
                run["max_deviation_px"] = deviation
                run["linear"] = deviation <= tolerance_px
            out.append(run)
            i = j + 1
    return out


def nan_rows_vs_fractional_boxes(videos: Sequence[VideoPoses]) -> dict[str, int]:
    """Cross all-NaN-confidence rows with rows whose bbox has a non-integer value.

    Detected boxes are stored as integers; an interpolated box usually is not.
    """
    nan_rows = frac_rows = both = 0
    for v in videos:
        all_nan = np.isnan(v.keypoints[..., 2]).sum(1) == v.keypoints.shape[1]
        frac = (v.bbox != np.round(v.bbox)).any(1)
        nan_rows += int(all_nan.sum())
        frac_rows += int(frac.sum())
        both += int((all_nan & frac).sum())
    return {"all_nan_rows": nan_rows, "fractional_box_rows": frac_rows, "both": both}


def axis_evidence(
    stored: npt.NDArray[np.float32], boxes: npt.NDArray[np.float32]
) -> dict[str, Any]:
    """Evidence for which stored keypoint column is vertical and how boxes are encoded.

    `stored` are keypoints exactly as in the file (N, 17, 3); `boxes` the four stored
    bbox numbers (N, 4). In an upright person the nose is above the ankles (smaller
    image row) and the shoulders are side by side.
    """
    nose_minus_ankles = stored[:, NOSE, :2] - stored[:, [L_ANKLE, R_ANKLE], :2].mean(1)
    shoulder_gap = np.abs(stored[:, L_SHOULDER, :2] - stored[:, R_SHOULDER, :2])
    b = boxes.astype(np.float64)
    evidence: dict[str, Any] = {
        "mean_nose_minus_ankles_per_column": np.nanmean(nose_minus_ankles, 0).tolist(),
        "mean_shoulder_gap_per_column": np.nanmean(shoulder_gap, 0).tolist(),
        "max_per_column": np.nanmax(stored[..., :2], axis=(0, 1)).tolist(),
        "bbox_max": b.max(0).tolist(),
    }
    for name, (cx, cy) in {"columns_xy": (0, 1), "columns_yx": (1, 0)}.items():
        x, y = stored[..., cx].astype(np.float64), stored[..., cy].astype(np.float64)
        xmax, ymax = np.nanmax(x, 1), np.nanmax(y, 1)
        evidence[name] = {
            "median_x2_minus_max_x_if_xyxy": float(np.median(b[:, 2] - xmax)),
            "median_y2_minus_max_y_if_xyxy": float(np.median(b[:, 3] - ymax)),
            "median_x_plus_w_minus_max_x_if_xywh": float(np.median(b[:, 0] + b[:, 2] - xmax)),
            "median_y_plus_h_minus_max_y_if_xywh": float(np.median(b[:, 1] + b[:, 3] - ymax)),
        }
    return evidence


def confidence_histograms(
    keypoints: npt.NDArray[np.float32], bins: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """(17, len(bins) - 1) counts of finite confidences per joint."""
    out = np.zeros((keypoints.shape[1], len(bins) - 1), np.int64)
    for j in range(keypoints.shape[1]):
        c = keypoints[:, j, 2]
        out[j] = np.histogram(c[np.isfinite(c)], bins=bins)[0]
    return out
