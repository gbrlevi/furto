"""Annotated videos of a scripts/ucf_items_eval.py run, one per video and zone source:
skeleton, hands coloured by hold, zones, contacts, releases, the alerts of both variants and
the annotated theft interval.

Usage: render_ucf_items.py RUN_DIR [SOURCE ...] (default: manual auto). UCF-Crime is 320 x 240,
so frames are enlarged `SCALE` times. Writes <RUN_DIR>/videos/<source>/<set>_<id>.mp4 at 15 fps;
output stays in runs/ (raw video never enters git).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from render_shop_items import BONES, captions
from shop_items import polygons
from ucf_items_eval import theft_spans

from tcc.pipeline.resample import frames_to_keep
from tcc.rules.shopping import hand_points
from tcc.utils.config import load_config

SCALE = 3
RED, ORANGE, WHITE = (0, 0, 255), (0, 140, 255), (255, 255, 255)


def main() -> None:
    run = Path(sys.argv[1])
    sources = sys.argv[2:] or ["manual", "auto"]
    c = json.loads((run / "meta.json").read_text(encoding="utf-8"))["config"]
    events = json.loads((run / "events.json").read_text(encoding="utf-8"))
    spans = theft_spans(c)
    videos = load_config(Path(c["videos_from"]))["videos"]
    for source in sources:
        zones = load_config(Path(c["zones"][source]))
        out_dir = run / "videos" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        for v in videos:
            vid = v["id"]
            name = next((s for s in ("official", "holdout") if vid in spans[s]), "normal")
            theft = spans[name].get(vid, []) if name != "normal" else []
            ev, scene = events[source][vid], zones.get(vid) or {}
            linked = ev["alerts_after_contact"]
            z = np.load(Path(c["hands"]) / f"{vid}.npz")
            cap = cv2.VideoCapture(v["path"])
            fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            keep = {int(s): q for q, s in enumerate(frames_to_keep(fps, total, c["fps"]))}
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * SCALE,
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * SCALE)  # fmt: skip
            writer = cv2.VideoWriter(str(out_dir / f"{name}_{vid}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), c["fps"], size)  # fmt: skip
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                q = keep.get(i)
                i += 1
                if q is None:
                    continue
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
                for kind, colour in (("shelf", (255, 200, 0)), ("checkout", (0, 200, 0))):
                    for poly in polygons(scene.get(kind)):
                        cv2.polylines(frame, [np.int32(np.asarray(poly) * SCALE)], True, colour, 2)
                for row in np.flatnonzero(z["frame_idx"] == q):
                    kp = z["keypoints"][row].astype(np.float64)
                    for a, b in BONES:
                        if min(kp[a, 2], kp[b, 2]) >= 0.3:
                            pa = tuple(map(int, kp[a, :2] * SCALE))
                            pb = tuple(map(int, kp[b, :2] * SCALE))
                            cv2.line(frame, pa, pb, WHITE, 2)
                    for hand, p in enumerate(hand_points(kp[None], c["contact"]["push"])[0]):
                        h = z["hold"][row, hand]
                        held = np.isfinite(h) and h >= c["hold"]["threshold"]
                        cv2.circle(frame, tuple(map(int, p * SCALE)), 10,
                                   RED if held else (160, 160, 160), 2)  # fmt: skip
                t = q / c["fps"]
                text = [(f"{name} {vid} [{source} zones] t={t:.1f}s", WHITE)]
                if any(a <= t <= b for a, b in theft):
                    text.append(("THEFT (annotated)", ORANGE))
                text += captions(ev, q, c["fps"])
                if any(a["frame"] <= q <= a["frame"] + 2 * c["fps"] for a in linked):
                    text.append(("ALERT after shelf contact", RED))
                for k, (s, colour) in enumerate(text):
                    cv2.putText(frame, s, (12, 30 + 30 * k), 0, 0.8, (0, 0, 0), 5)
                    cv2.putText(frame, s, (12, 30 + 30 * k), 0, 0.8, colour, 2)
                writer.write(frame)
            writer.release()
            cap.release()
            print(source, name, vid, flush=True)


if __name__ == "__main__":
    main()
