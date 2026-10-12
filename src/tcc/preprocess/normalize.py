"""Hip-centred, torso-scaled normalization of pose windows. Pure function, no IO.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from tcc.data.skeleton import HIPS, SHOULDERS

CENTERS = ("frame", "window")


def normalize(j: npt.NDArray[np.float32], conf_min: float, center: str) -> npt.NDArray[np.float32]:
    """Centre on the hips and divide by the torso length. `j`: (..., T, 17, 3).

    - Points with confidence below `conf_min` are missing: x, y and confidence become 0.
    - Centre: mid-hip per frame; if a hip is missing, the mean of the valid joints.
      With center="window", the mean of those centres over the window.
    - Scale: mean torso length (mid-shoulder to mid-hip) over the frames where both are
      valid; otherwise the vertical extent of the valid joints / 3 (fallback not in the
      Relatório); otherwise 1.
    """
    if center not in CENTERS:
        raise ValueError(f"center must be one of {CENTERS}")
    j = j.astype(np.float32, copy=True)
    xy, c = j[..., :2], j[..., 2]
    valid = c >= conf_min

    # Any: numpy's reductions are typed as scalar-or-array
    def mean_of(idx: list[int]) -> tuple[Any, Any]:
        return xy[..., idx, :].mean(-2), valid[..., idx].all(-1)

    hip, ok_hip = mean_of(HIPS)
    shoulder, ok_shoulder = mean_of(SHOULDERS)
    w = valid[..., None].astype(np.float32)
    mean_valid = (xy * w).sum(-2) / np.maximum(w.sum(-2), 1e-6)
    centre = np.where(ok_hip[..., None], hip, mean_valid)

    torso = np.linalg.norm(shoulder - hip, axis=-1)
    ok_torso = ok_hip & ok_shoulder
    n = ok_torso.sum(-1)
    # Any: the dtype changes along the fallbacks below (float32, float64)
    scale: npt.NDArray[Any] = np.where(n > 0, (torso * ok_torso).sum(-1) / np.maximum(n, 1), np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        ys = np.where(valid, xy[..., 1], np.nan)
        height = (np.nanmax(ys, axis=(-1, -2)) - np.nanmin(ys, axis=(-1, -2))) / 3.0
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, height)
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0)

    if center == "window":
        ok_c = ok_hip | valid.any(-1)
        cj = (centre * ok_c[..., None]).sum(-2) / np.maximum(ok_c.sum(-1), 1)[..., None]
        centre = np.broadcast_to(cj[..., None, :], centre.shape)

    xy = (xy - centre[..., None, :]) / scale[..., None, None, None]
    xy[~valid] = 0.0
    j[..., :2] = xy
    j[..., 2] = np.where(valid, c, 0.0)
    return j
