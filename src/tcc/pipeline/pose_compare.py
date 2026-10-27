"""Compare two pose estimates of the same people (session 3B). Pure functions, no IO.

The displacement of each joint is divided by the torso length of the reference pose, so it
lives in the same units as the normalized windows (hip-centred, torso-scaled), where the
gaussian noise of the data augmentation is defined.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from tcc.data.skeleton import HIPS, SHOULDERS

FloatArray = npt.NDArray[np.float32]


def normalized_displacement(a: FloatArray, b: FloatArray, conf_min: float) -> FloatArray:
    """(N, 17) distance between the joints of two (N, 17, 3) estimates divided by the torso
    of `a` (mid-shoulder to mid-hip). NaN where a joint is below `conf_min` in either
    estimate, or where `a` has one of the four torso joints below it."""
    ok = (a[..., 2] >= conf_min) & (b[..., 2] >= conf_min)
    torso_ok = (a[:, SHOULDERS + HIPS, 2] >= conf_min).all(1)
    shoulder = a[:, SHOULDERS, :2].mean(1)
    hip = a[:, HIPS, :2].mean(1)
    torso = np.linalg.norm(shoulder - hip, axis=-1)
    torso_ok &= torso > 0
    dist = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)
    out = dist / np.where(torso_ok, torso, 1.0)[:, None]
    return np.where(ok & torso_ok[:, None], out, np.nan).astype(np.float32)


def displacement_summary(d: FloatArray) -> dict[str, Any]:
    """Mean, median and upper percentiles of the finite displacements, with counts."""
    finite = d[np.isfinite(d)]
    without_torso = int(np.isnan(d).all(1).sum()) if len(d) else 0
    stats = (
        {
            "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "p90": float(np.percentile(finite, 90)),
            "p99": float(np.percentile(finite, 99)),
        }
        if len(finite)
        else {"mean": None, "median": None, "p90": None, "p99": None}
    )
    return {**stats, "points": int(len(finite)), "poses": int(len(d)),
            "poses_without_torso": without_torso}  # fmt: skip
