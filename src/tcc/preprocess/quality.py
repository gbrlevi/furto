"""Pose quality of each window and the quality gate (ADR-0014). Pure functions, no IO.

A window is scorable only if it passes every criterion whose threshold is given. The
thresholds come from the PoseLift training distribution (session 2D, task 3), never
from the demonstration videos or the test set.
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from tcc.data.skeleton import HIPS, SHOULDERS

FloatArray = npt.NDArray[np.float64]

# Criteria in reporting order; a window failing several gets the first as its reason.
CRITERIA = ("mean_confidence", "real_fraction", "torso_fraction", "box_height", "border_fraction")
# Lower bounds fail below the threshold (percentile 1 of training); upper bounds above it
# (percentile 99 of training).
UPPER_BOUNDED = ("border_fraction",)
SCORABLE = 0  # reason code; criterion CRITERIA[i] failing gives i + 1
REASONS = {SCORABLE: "scorable"} | {i + 1: name for i, name in enumerate(CRITERIA)}

TORSO = SHOULDERS + HIPS


def touches_border(
    boxes: npt.NDArray[np.float32], image_size: tuple[int, int], margin: float
) -> npt.NDArray[np.bool_]:
    """(..., 4) x1, y1, x2, y2 boxes within `margin` px of the (width, height) image edge."""
    w, h = image_size
    x1, y1, x2, y2 = np.moveaxis(boxes, -1, 0)
    out: npt.NDArray[np.bool_] = (
        (x1 <= margin) | (y1 <= margin) | (x2 >= w - margin) | (y2 >= h - margin)
    )
    return out


def window_quality(
    keypoints: npt.NDArray[np.float32],
    boxes: npt.NDArray[np.float32],
    real: npt.NDArray[np.bool_],
    image_size: tuple[int, int],
    conf_min: float,
    border_margin: float,
) -> dict[str, FloatArray]:
    """Quality measures of (N, T, 17, 3) windows before normalization, with their (N, T, 4)
    boxes and (N, T) real-detection masks; one value per window and criterion."""
    torso_ok = (keypoints[..., TORSO, 2] >= conf_min).sum(-1) >= 3
    return {
        "mean_confidence": keypoints[..., 2].mean((-1, -2)).astype(np.float64),
        "real_fraction": real.mean(-1, dtype=np.float64),
        "torso_fraction": torso_ok.mean(-1, dtype=np.float64),
        "box_height": (boxes[..., 3] - boxes[..., 1]).mean(-1, dtype=np.float64),
        "border_fraction": touches_border(boxes, image_size, border_margin).mean(
            -1, dtype=np.float64
        ),
    }


def torso_box_ratio(
    keypoints: npt.NDArray[np.float32], boxes: npt.NDArray[np.float32], conf_min: float
) -> FloatArray:
    """Torso length (mid-shoulder to mid-hip) over box height for each (..., 17, 3) pose;
    NaN unless the four torso joints are at or above `conf_min` and the box has height."""
    xy = keypoints[..., :2].astype(np.float64)
    torso = np.linalg.norm(xy[..., SHOULDERS, :].mean(-2) - xy[..., HIPS, :].mean(-2), axis=-1)
    height = (boxes[..., 3] - boxes[..., 1]).astype(np.float64)
    ok = (keypoints[..., TORSO, 2] >= conf_min).all(-1) & (height > 0)
    return np.where(ok, torso / np.where(height > 0, height, 1.0), np.nan)


def failures(
    metrics: Mapping[str, FloatArray], thresholds: Mapping[str, float]
) -> dict[str, npt.NDArray[np.bool_]]:
    """Which windows fail each criterion that has a threshold."""
    unknown = set(thresholds) - set(CRITERIA)
    if unknown:
        raise ValueError(f"unknown quality criteria {sorted(unknown)}")
    return {
        name: metrics[name] > thr if name in UPPER_BOUNDED else metrics[name] < thr
        for name, thr in thresholds.items()
    }


def gate_reasons(
    metrics: Mapping[str, FloatArray], thresholds: Mapping[str, float]
) -> npt.NDArray[np.int8]:
    """Reason code of each window: SCORABLE, or i + 1 for the first failing CRITERIA[i]."""
    n = len(next(iter(metrics.values())))
    reason = np.full(n, SCORABLE, np.int8)
    failed = failures(metrics, thresholds)
    for i, name in reversed(list(enumerate(CRITERIA))):
        if name in failed:
            reason[failed[name]] = i + 1
    return reason
