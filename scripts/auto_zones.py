"""Shelf and checkout zones found automatically on each video of a fixed camera.

Usage: auto_zones.py CONFIG. For each video, the background is the per-pixel median of
`background_frames` frames spread over the video (people who move are removed); OWLv2 looks for
the text queries of each zone kind on it, and `tcc.rules.zone_proposals.propose` turns the
boxes into polygons. Writes zones.yaml ({video: {shelf: [...], checkout: [...]}}, the format
of configs/zones/) and a background image with the zones drawn for each video in the run.
"""

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from tcc.rules.zone_proposals import propose
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run

COLOURS = {"shelf": (255, 200, 0), "checkout": (0, 200, 0)}


def background(path: str, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).round().astype(int).tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def detect(image: np.ndarray, c: dict[str, Any], proc: Any, model: Any) -> dict[str, Any]:
    kinds = [k for k, qs in c["queries"].items() for _ in qs]
    texts = [q for qs in c["queries"].values() for q in qs]
    h, w = image.shape[:2]
    x = proc(text=[texts], images=Image.fromarray(image[:, :, ::-1]), return_tensors="pt")
    with torch.no_grad():
        out = model(**x.to(model.device))
    side = max(w, h)  # OWLv2 pads to a square at the bottom and right
    r = proc.post_process_grounded_object_detection(
        outputs=out, threshold=c["threshold"], target_sizes=torch.tensor([[side, side]])
    )[0]
    return propose(
        r["boxes"].cpu().numpy().astype(np.float64),
        r["scores"].cpu().numpy().astype(np.float64),
        r["labels"].cpu().numpy().astype(np.int64),
        kinds, w, h, c["nms_iou"], c["max_area"], c["max_boxes"],
    )  # fmt: skip


def main() -> None:
    c = load_config(Path(sys.argv[1]))
    run = start_run(c["name"], config=c, seeds={})
    proc = Owlv2Processor.from_pretrained(c["model"])
    model = Owlv2ForObjectDetection.from_pretrained(c["model"]).to(c["device"]).eval()
    videos = load_config(Path(c["videos_from"]))["videos"]
    zones = {}
    for v in videos:
        image = background(v["path"], c["background_frames"])
        zones[v["id"]] = found = detect(image, c, proc, model)
        for kind, polygons in found.items():
            for poly in polygons:
                cv2.polylines(image, [np.int32(poly)], True, COLOURS[kind], 2)
        cv2.imwrite(str(run / f"{v['id']}_zones.jpg"), image)
        print(v["id"], {k: len(p) for k, p in found.items()}, flush=True)
    (run / "zones.yaml").write_text(yaml.safe_dump(zones, sort_keys=True), encoding="utf-8")
    print("run", run)


if __name__ == "__main__":
    main()
