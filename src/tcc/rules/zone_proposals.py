"""Shelf and checkout zones proposed from the boxes an open-vocabulary detector found on one
background image of a fixed camera (scripts/auto_zones.py). Pure geometry: no model here.

Each query of the detector belongs to a zone kind ("shelf", "checkout"). Per kind, boxes are
clipped to the image, boxes larger than `max_area` of the image are dropped (a box over the
whole scene would put every hand in the zone), overlapping boxes are suppressed and the
`max_boxes` best are kept as rectangles.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]


def box_iou(a: Floats, b: Floats) -> Floats:
    """(len(a), len(b)) intersection over union of x1y1x2y2 boxes."""
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(hi - lo, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.nan_to_num(inter / (area_a[:, None] + area_b[None, :] - inter)))


def nms(boxes: Floats, scores: Floats, iou: float) -> list[int]:
    """Indices kept by greedy non-maximum suppression, best score first."""
    kept: list[int] = []
    for i in np.argsort(-scores, kind="stable").tolist():
        if not kept or box_iou(boxes[[i]], boxes[kept]).max() <= iou:
            kept.append(i)
    return kept


def propose(
    boxes: Floats,
    scores: Floats,
    labels: npt.NDArray[np.int64],
    kinds: Sequence[str],
    width: int,
    height: int,
    nms_iou: float,
    max_area: float,
    max_boxes: Mapping[str, int],
) -> dict[str, list[list[list[float]]]]:
    """Polygons per zone kind; `labels` index `kinds` (the kind of each detector query)."""
    clipped = np.clip(boxes, 0, [width, height, width, height]).astype(np.float64)
    area = np.prod(clipped[:, 2:] - clipped[:, :2], axis=1)
    ok = (area > 0) & (area <= max_area * width * height)
    out: dict[str, list[list[list[float]]]] = {}
    for kind, limit in max_boxes.items():
        idx = np.flatnonzero(ok & np.isin(labels, [i for i, k in enumerate(kinds) if k == kind]))
        keep = [int(idx[i]) for i in nms(clipped[idx], scores[idx], nms_iou)][:limit]
        out[kind] = [
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            for x1, y1, x2, y2 in np.round(clipped[keep], 1).tolist()
        ]
    return out
