"""Item-level shop rules: hand contact with a shelf, an item taken, where the item was let
go, and whether the items that left the shop were paid for.

Inputs are the pose records of one video (ADR-0004 contract), zone polygons in the same
pixels, and a per-hand probability that the hand holds a product (from any image model).
Hand 0 is the left hand and hand 1 the right one (COCO-17 elbow and wrist).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tcc.data.schema import VideoPoses
from tcc.rules.zones import Polygon, in_polygon, runs

ELBOWS, WRISTS = (7, 8), (9, 10)
SHOULDERS, HIPS = (5, 6), (11, 12)


@dataclass(frozen=True)
class Contact:
    track: int
    hand: int
    start: int  # first frame with the hand in the shelf zone
    end: int  # last one


@dataclass(frozen=True)
class Take:
    track: int
    hand: int
    frame: int  # end of the contact
    before: float  # median hold probability before the contact
    after: float  # and after it


@dataclass(frozen=True)
class Release:
    track: int
    hand: int
    held_from: int  # first frame of the hold run
    frame: int  # last frame the hand was seen holding
    where: str  # a zone name, "body" or "unknown"


@dataclass(frozen=True)
class ShopAlert:
    track: int
    frame: int
    kind: str  # "concealment" or "unpaid_exit"
    items: int  # items taken and not put back (unpaid_exit) or 1 (concealment)


def hand_points(keypoints: npt.NDArray[np.float32], push: float) -> npt.NDArray[np.float64]:
    """(N, 2, 2) hand centres: each wrist pushed `push` forearm lengths past the elbow line."""
    elbow = keypoints[:, ELBOWS, :2].astype(np.float64)
    wrist = keypoints[:, WRISTS, :2].astype(np.float64)
    return wrist + push * (wrist - elbow)


def hand_conf(keypoints: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """(N, 2) confidence of a hand point: the lower of its wrist and elbow."""
    return np.minimum(keypoints[:, ELBOWS, 2], keypoints[:, WRISTS, 2]).astype(np.float64)


def hands_in_zone(
    points: npt.NDArray[np.float64],
    conf: npt.NDArray[np.float64],
    polygon: Polygon,
    min_conf: float,
) -> npt.NDArray[np.bool_]:
    """(N, 2): confident hand inside the polygon."""
    inside = in_polygon(points.reshape(-1, 2), polygon).reshape(points.shape[:2])
    return np.asarray(inside & (conf >= min_conf), bool)


def contacts(
    video: VideoPoses, inside: npt.NDArray[np.bool_], max_gap: int, min_frames: int
) -> list[Contact]:
    """Runs of each hand of each track inside the shelf zone, ordered by start."""
    out = []
    for track in np.unique(video.track_id).tolist():
        own = video.track_id == track
        for hand in (0, 1):
            frames = np.unique(video.frame_idx[own & inside[:, hand]])
            out += [Contact(track, hand, a, b) for a, b in runs(frames, max_gap, min_frames)]
    return sorted(out, key=lambda c: (c.start, c.track, c.hand))


def _median_hold(
    video: VideoPoses, hold: npt.NDArray[np.float64], c: Contact, lo: int, hi: int
) -> float:
    rows = (video.track_id == c.track) & (video.frame_idx >= lo) & (video.frame_idx <= hi)
    values = hold[rows, c.hand]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def takes(
    video: VideoPoses,
    contacts_: Sequence[Contact],
    hold: npt.NDArray[np.float64],
    window: int,
) -> list[Take]:
    """Hold probability of the same hand in the `window` frames before and after each
    contact; the caller decides with a threshold whether an item came out (or went back)."""
    return [
        Take(
            c.track,
            c.hand,
            c.end,
            _median_hold(video, hold, c, c.start - window, c.start - 1),
            _median_hold(video, hold, c, c.end + 1, c.end + window),
        )
        for c in contacts_
    ]


def near_body(
    points: npt.NDArray[np.float64], keypoints: npt.NDArray[np.float32], radius: float
) -> npt.NDArray[np.bool_]:
    """(N, 2): hand within `radius` torso lengths of the torso centre (mean of shoulders and
    hips). Rows without a torso are never near it."""
    shoulders = keypoints[:, SHOULDERS, :2].mean(1).astype(np.float64)
    hips = keypoints[:, HIPS, :2].mean(1).astype(np.float64)
    torso = np.linalg.norm(shoulders - hips, axis=1)
    centre = (shoulders + hips) / 2
    dist = np.linalg.norm(points - centre[:, None, :], axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray((torso[:, None] > 0) & (dist <= radius * torso[:, None]), bool)


def releases(
    video: VideoPoses,
    hold: npt.NDArray[np.float64],
    threshold: float,
    places: Mapping[str, npt.NDArray[np.bool_]],
    body: npt.NDArray[np.bool_],
    max_gap: int,
    min_frames: int,
    look: int,
) -> list[Release]:
    """End of every run in which a hand held something (hold >= threshold). Where it was let
    go: the first zone of `places` (in order) the hand touched in the `look` frames around
    the end, else "body" when the hand was near the torso then, else "unknown". A run that
    reaches the last frame of its track is not a release."""
    out = []
    for track in np.unique(video.track_id).tolist():
        own = video.track_id == track
        last = int(video.frame_idx[own].max())
        for hand in (0, 1):
            held = own & (np.nan_to_num(hold[:, hand], nan=-np.inf) >= threshold)
            for a, b in runs(np.unique(video.frame_idx[held]), max_gap, min_frames):
                if b + max_gap >= last:
                    continue
                around = own & (video.frame_idx >= b - look) & (video.frame_idx <= b + look)
                where = next((n for n, m in places.items() if m[around, hand].any()), None)
                if where is None:
                    where = "body" if body[around, hand].any() else "unknown"
                out.append(Release(track, hand, a, b, where))
    return sorted(out, key=lambda r: (r.frame, r.track, r.hand))


def shop_alerts(
    taken: Sequence[Take],
    released: Sequence[Release],
    exits: Mapping[int, Sequence[int]],
    payments: Sequence[tuple[int, int | None]],
    threshold: float,
    grace: int,
) -> list[ShopAlert]:
    """Concealment: every release on the body. Unpaid exit: at each exit of a track, the items
    it took since its previous exit (contacts that ended with the hand holding when it did not
    before) minus the ones it put back (contacts the other way round) exceed the items paid
    between its first take and the exit plus `grace`. A payment with no item count pays for
    everything."""
    alerts = [ShopAlert(r.track, r.frame, "concealment", 1) for r in released if r.where == "body"]
    for track, frames in exits.items():
        own = [t for t in taken if t.track == track]
        since = -1
        for e in sorted(frames):
            window = [t for t in own if since < t.frame <= e]
            got = [t for t in window if t.after >= threshold and not t.before >= threshold]
            back = [t for t in window if t.before >= threshold and not t.after >= threshold]
            items = len(got) - len(back)
            if items > 0:
                first = min(t.frame for t in got)
                paid = [n for f, n in payments if first <= f <= e + grace]
                if not (None in paid or sum(n for n in paid if n is not None) >= items):
                    alerts.append(ShopAlert(track, e, "unpaid_exit", items))
            since = e
    return sorted(alerts, key=lambda a: (a.frame, a.track, a.kind))
