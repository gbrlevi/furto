"""Zone rule for an unattended shop: a person who stayed at a shelf and then reached the exit
without stopping at the checkout (and with no payment recorded meanwhile) raises an alert.

Zones are polygons in image pixels drawn per camera. A person's position is the midpoint of
the ankles when both are confident, otherwise the bottom centre of the box.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tcc.data.schema import VideoPoses

LEFT_ANKLE, RIGHT_ANKLE = 15, 16
Polygon = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class Visit:
    track: int
    zone: str
    start: int  # first frame inside
    end: int  # last frame inside


@dataclass(frozen=True)
class ExitAlert:
    track: int
    frame: int  # first frame at the exit
    shelf_frames: int  # frames spent at shelves since the track appeared or last exited
    first_shelf: int


@dataclass(frozen=True)
class ExitRule:
    min_shelf_frames: int  # time at shelves that counts as having picked something
    min_checkout_frames: int  # time at the checkout that counts as having paid
    payment_grace_frames: int  # a payment up to this long after reaching the exit still counts


def foot_points(
    bbox: npt.NDArray[np.float32], keypoints: npt.NDArray[np.float32], min_conf: float
) -> npt.NDArray[np.float64]:
    """(N, 2) ground position of each row."""
    ankles = keypoints[:, [LEFT_ANKLE, RIGHT_ANKLE]]
    both = np.asarray((ankles[:, :, 2] >= min_conf).all(1), bool)
    box = np.column_stack([(bbox[:, 0] + bbox[:, 2]) / 2, bbox[:, 3]]).astype(np.float64)
    feet = ankles[:, :, :2].mean(1).astype(np.float64)
    return np.where(both[:, None], feet, box)


def in_polygon(points: npt.NDArray[np.float64], polygon: Polygon) -> npt.NDArray[np.bool_]:
    """Even-odd ray casting; points on an edge may fall either way."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), bool)
    p = list(polygon)
    for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1], strict=True):
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < at)
    return inside


def runs(frames: npt.NDArray[np.int64], max_gap: int, min_frames: int) -> list[tuple[int, int]]:
    """(start, end) of runs of sorted frames, joining gaps of up to `max_gap` frames and
    keeping runs that span at least `min_frames` frames."""
    if len(frames) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) > max_gap + 1)
    starts = np.r_[frames[0], frames[breaks + 1]]
    ends = np.r_[frames[breaks], frames[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True) if b - a + 1 >= min_frames]


def track_visits(
    video: VideoPoses,
    zones: Mapping[str, Polygon],
    min_conf: float,
    max_gap: int,
    min_frames: int,
) -> list[Visit]:
    """Visits of every track to every zone, ordered by start frame."""
    points = foot_points(video.bbox, video.keypoints, min_conf)
    visits = []
    for name, polygon in zones.items():
        inside = in_polygon(points, polygon)
        for track in np.unique(video.track_id):
            rows = (video.track_id == track) & inside
            frames = np.unique(video.frame_idx[rows])
            for a, b in runs(frames, max_gap, min_frames):
                visits.append(Visit(int(track), name, a, b))
    return sorted(visits, key=lambda v: (v.start, v.track, v.zone))


def unpaid_exits(
    visits: Sequence[Visit],
    payments: Sequence[int],
    rule: ExitRule,
    shelf: str = "shelf",
    checkout: str = "checkout",
    exit_zone: str = "exit",
) -> list[ExitAlert]:
    """One alert per exit of a track that spent `min_shelf_frames` at shelves since it
    appeared (or since its previous exit) without `min_checkout_frames` at the checkout and
    without any payment between its first shelf frame and the exit plus the grace."""
    alerts = []
    for track in sorted({v.track for v in visits}):
        own = [v for v in visits if v.track == track]
        since = -1
        for e in (v for v in own if v.zone == exit_zone):
            window = [v for v in own if since < v.start < e.start]
            shelves = [v for v in window if v.zone == shelf]
            took = sum(v.end - v.start + 1 for v in shelves)
            if took >= rule.min_shelf_frames:
                first = min(v.start for v in shelves)
                at_checkout = sum(v.end - v.start + 1 for v in window if v.zone == checkout)
                paid = any(first <= p <= e.start + rule.payment_grace_frames for p in payments)
                if at_checkout < rule.min_checkout_frames and not paid:
                    alerts.append(ExitAlert(track, e.start, took, first))
            since = e.end
    return alerts
