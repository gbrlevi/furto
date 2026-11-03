"""The item pipeline on videos with hand features: shelf contacts, takes, where each held item
was let go, and the concealment / unpaid-exit alerts of `tcc.rules.shopping`.

Usage: shop_items.py CONFIG. Hand features come from scripts/hand_hold_scores.py; the hold
probability is the zero-shot one or a logistic probe saved by scripts/merl_stages.py.
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
    logit = z["emb"].reshape(-1, d).astype(np.float64) @ w["coef"].ravel() + float(w["intercept"])
    return np.where(np.isfinite(hold), 1 / (1 + np.exp(-logit.reshape(n, 2))), np.nan)


def run_video(item: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    z = np.load(Path(c["hands"]) / f"{item['id']}.npz")
    video = VideoPoses(
        video_id=item["id"], camera="synthetic", n_frames=int(z["n_frames"]),
        frame_idx=z["frame_idx"].astype(np.int64), track_id=z["track_id"].astype(np.int64),
        bbox=z["bbox"], keypoints=z["keypoints"], frame_labels=None,
    )  # fmt: skip
    scene, ct, rl = c["scenes"][item["scene"]], c["contact"], c["release"]
    hold, thr = hold_of(z, c), c["hold"]["threshold"]
    points, conf = hand_points(video.keypoints, ct["push"]), hand_conf(video.keypoints)
    shelf = np.zeros((len(video.frame_idx), 2), bool)
    for polygon in scene["shelf"]:
        shelf |= hands_in_zone(points, conf, polygon, ct["min_conf"])
    found = contacts(video, shelf, ct["max_gap"], ct["min_frames"])
    taken = takes(video, found, hold, c["take"]["window"])
    places = {
        "shelf": shelf,
        "checkout": hands_in_zone(points, conf, scene["checkout"], ct["min_conf"]),
    }
    body = near_body(points, video.keypoints, rl["body_radius"])
    released = releases(video, hold, thr, places, body, rl["max_gap"], rl["min_frames"], rl["look"])
    exits = {
        int(t): [int(video.frame_idx[video.track_id == t].max())] for t in np.unique(video.track_id)
    }
    payments = [(round(s * c["fps"]), int(n)) for s, n in item["payments"]]
    grace = round(c["payment_grace_seconds"] * c["fps"])
    alerts = shop_alerts(taken, released, exits, payments, thr, grace)
    concealment = any(a.kind == "concealment" for a in alerts)
    return {
        "expected": item["expected"],
        "concealment_alert": concealment,
        "as_expected": concealment == (item["expected"] == "concealment"),
        "contacts": [asdict(x) for x in found],
        "takes": [asdict(x) for x in taken],
        "releases": [asdict(x) for x in released],
        "alerts": [asdict(x) for x in alerts],
        "hold_median": float(np.nanmedian(hold)),
    }


def main() -> None:
    c = load_config(Path(sys.argv[1]))
    run = start_run("shop_items", config=c, seeds={})
    events = {item["id"]: run_video(item, c) for item in c["videos"]}
    summary = {
        k: {
            "expected": v["expected"],
            "concealment_alert": v["concealment_alert"],
            "as_expected": v["as_expected"],
            "releases": [(r["frame"], r["where"]) for r in v["releases"]],
            "takes": len([t for t in v["takes"] if t["after"] >= c["hold"]["threshold"]]),
            "alerts": [(a["frame"], a["kind"], a["items"]) for a in v["alerts"]],
        }
        for k, v in events.items()
    }
    (run / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for k, v in summary.items():
        print(k, v)
    print("as expected", sum(v["as_expected"] for v in summary.values()), "of", len(summary))
    print("run", run)


if __name__ == "__main__":
    main()
