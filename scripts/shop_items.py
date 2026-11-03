"""The item pipeline on videos with hand features: shelf contacts, takes, where each held item
was let go, and the concealment / unpaid-exit alerts of `tcc.rules.shopping`.

Usage: shop_items.py CONFIG. Hand features come from scripts/hand_hold_scores.py; the hold
probability is the zero-shot one or a logistic probe saved by scripts/merl_stages.py. Zones
are the `scenes` of the config or, with `zones`, a file keyed by video (configs/zones/ or the
zones.yaml of scripts/auto_zones.py). Concealment is reported twice: any item let go on the
body (main) and only an item whose hold began within `link_seconds` of a shelf contact.
Writes events.json (every contact, take, release and alert per video) and summary.json.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tcc.data.schema import VideoPoses
from tcc.rules.shopping import (
    after_contact,
    contacts,
    hand_conf,
    hand_points,
    hands_in_zone,
    near_body,
    releases,
    shop_alerts,
    takes,
)
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def hold_of(z: Any, c: dict[str, Any]) -> np.ndarray:  # Any: NpzFile
    hold = z["hold"].astype(np.float64)
    if c["hold"]["source"] == "zero_shot":
        return hold
    w = np.load(c["hold"]["probe"])
    n, _, d = z["emb"].shape
    logit = z["emb"].reshape(-1, d).astype(np.float64) @ w["coef"].ravel() + w["intercept"].item()
    return np.where(np.isfinite(hold), 1 / (1 + np.exp(-logit.reshape(n, 2))), np.nan)


def load_video(z: Any, vid: str, camera: str) -> VideoPoses:  # Any: NpzFile
    return VideoPoses(
        video_id=vid, camera=camera, n_frames=int(z["n_frames"]),
        frame_idx=z["frame_idx"].astype(np.int64), track_id=z["track_id"].astype(np.int64),
        bbox=z["bbox"], keypoints=z["keypoints"], frame_labels=None,
    )  # fmt: skip


def polygons(zone: Any) -> list[Any]:
    """A zone given as one polygon or as a list of polygons, as a list of polygons."""
    if not zone:
        return []
    return [zone] if isinstance(zone[0][0], int | float) else list(zone)


def in_zones(points: np.ndarray, conf: np.ndarray, zone: Any, min_conf: float) -> np.ndarray:
    inside = np.zeros(points.shape[:2], bool)
    for polygon in polygons(zone):
        inside |= hands_in_zone(points, conf, polygon, min_conf)
    return inside


def run_pipeline(
    video: VideoPoses,
    hold: np.ndarray,
    scene: dict[str, Any],
    c: dict[str, Any],
    payments: list[tuple[int, int | None]],
) -> dict[str, Any]:
    ct, rl, thr = c["contact"], c["release"], c["hold"]["threshold"]
    points, conf = hand_points(video.keypoints, ct["push"]), hand_conf(video.keypoints)
    shelf = in_zones(points, conf, scene.get("shelf"), ct["min_conf"])
    found = contacts(video, shelf, ct["max_gap"], ct["min_frames"])
    taken = takes(video, found, hold, c["take"]["window"])
    places = {
        "shelf": shelf,
        "checkout": in_zones(points, conf, scene.get("checkout"), ct["min_conf"]),
    }
    body = near_body(points, video.keypoints, rl["body_radius"])
    released = releases(video, hold, thr, places, body, rl["max_gap"], rl["min_frames"], rl["look"])
    exits = {
        int(t): [int(video.frame_idx[video.track_id == t].max())] for t in np.unique(video.track_id)
    }
    grace = round(c["payment_grace_seconds"] * c["fps"])
    alerts = shop_alerts(taken, released, exits, payments, thr, grace)
    linked = after_contact(released, found, round(c["link_seconds"] * c["fps"]))
    return {
        "contacts": [asdict(x) for x in found],
        "takes": [asdict(x) for x in taken],
        "releases": [asdict(x) for x in released],
        "alerts": [asdict(x) for x in alerts],
        "alerts_after_contact": [
            {"track": r.track, "frame": r.frame, "kind": "concealment", "items": 1}
            for r in linked
            if r.where == "body"
        ],
        "hold_median": float(np.nanmedian(hold)) if np.isfinite(hold).any() else None,
    }


def scenes_of(c: dict[str, Any]) -> dict[str, Any]:
    if c.get("zones"):
        return dict(load_config(Path(c["zones"])))
    return {item["id"]: c["scenes"][item["scene"]] for item in c["videos"]}


def main() -> None:
    c = load_config(Path(sys.argv[1]))
    run = start_run(c.get("name", "shop_items"), config=c, seeds={})
    scenes = scenes_of(c)
    events = {}
    for item in c["videos"]:
        z = np.load(Path(c["hands"]) / f"{item['id']}.npz")
        payments = [(round(s * c["fps"]), int(n)) for s, n in item["payments"]]
        ev = run_pipeline(load_video(z, item["id"], "synthetic"), hold_of(z, c),
                          scenes[item["id"]], c, payments)  # fmt: skip
        want = item["expected"] == "concealment"
        main_alert = any(a["kind"] == "concealment" for a in ev["alerts"])
        linked_alert = bool(ev["alerts_after_contact"])
        events[item["id"]] = {
            "expected": item["expected"],
            "concealment_alert": main_alert,
            "as_expected": main_alert == want,
            "concealment_after_contact": linked_alert,
            "as_expected_after_contact": linked_alert == want,
            **ev,
        }
    summary = {
        k: {
            "expected": v["expected"],
            "as_expected": v["as_expected"],
            "as_expected_after_contact": v["as_expected_after_contact"],
            "contacts": len(v["contacts"]),
            "releases": [(r["frame"], r["where"]) for r in v["releases"]],
            "alerts": [(a["frame"], a["kind"], a["items"]) for a in v["alerts"]],
            "after_contact": [a["frame"] for a in v["alerts_after_contact"]],
        }
        for k, v in events.items()
    }
    (run / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for k, v in summary.items():
        print(k, v)
    for key in ("as_expected", "as_expected_after_contact"):
        print(key, sum(v[key] for v in summary.values()), "of", len(summary))
    print("run", run)


if __name__ == "__main__":
    main()
