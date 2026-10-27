"""Pose distributions to compare our extraction with PoseLift (session 3B, task 5).

Pure functions on contract arrays (x, y in pixels, confidence). A joint counts when its
confidence is >= `conf_min`, as in the treatment. Bone lengths are divided by the torso
(mid-shoulder to mid-hip), so the measures do not depend on the image resolution.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from tcc.data.skeleton import EDGES, HIPS, SHOULDERS

FloatArray = npt.NDArray[np.float32]
NOSE, ANKLES = 0, (15, 16)
__all__ = ["EDGES", "bone_ratios", "box_extent_ratio", "joint_confidence"]


def bone_ratios(keypoints: FloatArray, conf_min: float) -> FloatArray:
    """(N, 19) length of each bone of `EDGES` over the torso length of the same pose; NaN
    when either joint or any of the four torso joints is below `conf_min`."""
    ok = keypoints[..., 2] >= conf_min
    torso_ok = ok[:, SHOULDERS + HIPS].all(1)
    xy = keypoints[..., :2]
    torso = np.linalg.norm(xy[:, SHOULDERS].mean(1) - xy[:, HIPS].mean(1), axis=-1)
    torso_ok &= torso > 0
    a, b = np.array(EDGES).T
    length = np.linalg.norm(xy[:, a] - xy[:, b], axis=-1)
    ratio = length / np.where(torso_ok, torso, 1.0)[:, None]
    return np.where(ok[:, a] & ok[:, b] & torso_ok[:, None], ratio, np.nan).astype(np.float32)


def box_extent_ratio(boxes: FloatArray, keypoints: FloatArray, conf_min: float) -> FloatArray:
    """(N,) box height over the vertical extent of the joints at or above `conf_min`, for
    people with the nose and both ankles at or above it; NaN for the others."""
    ok = keypoints[..., 2] >= conf_min
    visible = ok[:, NOSE] & ok[:, list(ANKLES)].all(1)
    out = np.full(len(boxes), np.nan, np.float32)
    rows = np.flatnonzero(visible)
    if len(rows) == 0:
        return out
    y = np.where(ok[rows], keypoints[rows, :, 1], np.nan)
    extent = np.nanmax(y, 1) - np.nanmin(y, 1)  # the nose and ankles are never NaN here
    height = boxes[rows, 3] - boxes[rows, 1]
    out[rows] = np.where(extent > 0, height / np.where(extent > 0, extent, 1.0), np.nan)
    return out


def joint_confidence(keypoints: FloatArray) -> dict[int, dict[str, Any]]:
    """Mean, median and quartiles of the confidence of each joint over all rows."""
    c = keypoints[..., 2]
    out = {}
    for j in range(c.shape[1]):
        v = c[:, j][np.isfinite(c[:, j])]
        out[j] = {
            "mean": float(v.mean()) if len(v) else None,
            "median": float(np.median(v)) if len(v) else None,
            "p25": float(np.percentile(v, 25)) if len(v) else None,
            "p75": float(np.percentile(v, 75)) if len(v) else None,
            "rows": int(len(v)),
        }
    return out
