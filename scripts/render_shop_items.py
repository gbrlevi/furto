"""Annotated videos of an item-pipeline run: skeleton, hands coloured by hold (red holding,
grey empty), zones, and the contacts, releases and alerts of events.json as captions.

Usage: render_shop_items.py RUN_DIR. Reads the run's config and events.json; writes
<RUN_DIR>/<id>_items.mp4 at 15 fps. Output stays in runs/ (raw video never enters git).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tcc.pipeline.resample import frames_to_keep
from tcc.rules.shopping import hand_points
from tcc.utils.config import load_config

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13),
         (13, 15), (12, 14), (14, 16)]  # fmt: skip


def captions(ev: dict, q: int, fps: float) -> list[tuple[str, tuple[int, int, int]]]:
    out = []
    for c in ev["contacts"]:
        if c["start"] <= q <= c["end"]:
            out.append((f"hand {c['hand']} at shelf", (255, 200, 0)))
    for r in ev["releases"]:
        if r["frame"] <= q <= r["frame"] + fps:
            colour = (0, 0, 255) if r["where"] == "body" else (0, 200, 0)
            out.append((f"let go: {r['where']}", colour))
    for a in ev["alerts"]:
        if a["frame"] <= q <= a["frame"] + 2 * fps:
            out.append((f"ALERT {a['kind']} ({a['items']})", (0, 0, 255)))
    return out


def main() -> None:
    run = Path(sys.argv[1])
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    c = meta["config"]
    events = json.loads((run / "events.json").read_text(encoding="utf-8"))
    raw = load_config(Path("configs/hand_hold_synthetic.yaml"))
    for item in c["videos"]:
        vid, ev = item["id"], events[item["id"]]
        z = np.load(Path(c["hands"]) / f"{vid}.npz")
        hold = z["hold"]
        cap = cv2.VideoCapture(str(Path(raw["videos"]) / raw["video_name"].format(vid)))
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keep = {int(s): q for q, s in enumerate(frames_to_keep(fps, total, c["fps"]))}
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        writer = cv2.VideoWriter(str(run / f"{vid}_items.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 c["fps"], (width, height))  # fmt: skip
        scene = c["scenes"][item["scene"]]
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            q = keep.get(i)
            i += 1
            if q is None:
                continue
            for poly in scene["shelf"]:
                cv2.polylines(frame, [np.int32(poly)], True, (255, 200, 0), 1)
            cv2.polylines(frame, [np.int32(scene["checkout"])], True, (0, 200, 0), 1)
            for row in np.flatnonzero(z["frame_idx"] == q):
                kp = z["keypoints"][row]
                for a, b in BONES:
                    if min(kp[a, 2], kp[b, 2]) >= 0.3:
                        pa, pb = tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2]))
                        cv2.line(frame, pa, pb, (255, 255, 255), 2)
                for hand, p in enumerate(hand_points(kp[None], c["contact"]["push"])[0]):
                    h = hold[row, hand]
                    held = np.isfinite(h) and h >= c["hold"]["threshold"]
                    colour = (0, 0, 255) if held else (160, 160, 160)
                    cv2.circle(frame, tuple(map(int, p)), 12, colour, 3)
            text = [(f"{vid}  t={q / c['fps']:.1f}s", (255, 255, 255))]
            for k, (t, colour) in enumerate(text + captions(ev, q, c["fps"])):
                cv2.putText(frame, t, (20, 40 + 36 * k), 0, 1.0, (0, 0, 0), 5)
                cv2.putText(frame, t, (20, 40 + 36 * k), 0, 1.0, colour, 2)
            writer.write(frame)
        writer.release()
        cap.release()
        print(vid, flush=True)


if __name__ == "__main__":
    main()
