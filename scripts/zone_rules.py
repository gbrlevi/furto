"""Unpaid-exit alerts from the poses of one camera and the zones drawn for it.

Usage: zone_rules.py CONFIG [--draw]. With --draw, only writes a frame of the video with the
zones and a 100 px grid, to help place the polygons. Poses come from scripts/extract_pose.py
(15 fps); payments, if given, are a CSV with a `seconds` column (time of each sale).
"""

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from tcc.pipeline.scoring import load_extracted
from tcc.rules.zones import ExitRule, track_visits, unpaid_exits
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run

COLOURS = {"shelf": (0, 200, 255), "checkout": (0, 200, 0), "exit": (0, 0, 255)}


def draw(config: dict, out: Path) -> None:
    capture = cv2.VideoCapture(config["video"])
    capture.set(cv2.CAP_PROP_POS_FRAMES, config.get("draw_frame", 0))
    frame = capture.read()[1]
    h, w = frame.shape[:2]
    s = w / config["pose_size"][0]  # zones and grid are in pose pixels
    for x in range(0, int(w / s), 100):
        cv2.line(frame, (int(x * s), 0), (int(x * s), h), (80, 80, 80), 1)
        cv2.putText(frame, str(x), (int(x * s) + 2, 12), 0, 0.4, (255, 255, 255), 1)
    for y in range(0, int(h / s), 100):
        cv2.line(frame, (0, int(y * s)), (w, int(y * s)), (80, 80, 80), 1)
        cv2.putText(frame, str(y), (2, int(y * s) - 2), 0, 0.4, (255, 255, 255), 1)
    for name, polygon in config["zones"].items():
        pts = (np.asarray(polygon) * s).astype(np.int32)
        cv2.polylines(frame, [pts], True, COLOURS.get(name, (255, 255, 255)), 2)
        cv2.putText(frame, name, tuple(pts[0]), 0, 0.6, COLOURS.get(name, (255, 255, 255)), 2)
    cv2.imwrite(str(out), frame)


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    run = start_run("zone_rules", config=config, seeds={})
    if "--draw" in sys.argv:
        draw(config, run / "zones.png")
        print("run", run)
        return
    fps = config["fps"]
    video = load_extracted(Path(config["poses"]), Path(config["video"]).stem, config["camera"])
    t = config["tracking"]
    visits = track_visits(
        video,
        config["zones"],
        t["min_ankle_conf"],
        round(t["max_gap_seconds"] * fps),
        round(t["min_visit_seconds"] * fps),
    )
    r = config["rule"]
    rule = ExitRule(
        round(r["min_shelf_seconds"] * fps),
        round(r["min_checkout_seconds"] * fps),
        round(r["payment_grace_seconds"] * fps),
    )
    payments = []
    if config.get("payments"):
        with open(config["payments"], newline="", encoding="utf-8") as f:
            payments = [round(float(row["seconds"]) * fps) for row in csv.DictReader(f)]
    found = unpaid_exits(visits, payments, rule)
    result = {
        "visits": [asdict(v) for v in visits],
        "alerts": [{**asdict(a), "seconds": a.frame / fps} for a in found],
    }
    (run / "alerts.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    draw(config, run / "zones.png")
    print(
        len(visits),
        "visits;",
        len(found),
        "unpaid exits:",
        [a["seconds"] for a in result["alerts"]],
    )
    print("run", run)


if __name__ == "__main__":
    main()
