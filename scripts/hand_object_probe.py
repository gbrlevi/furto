"""Does a pretrained image model see a product in the hand? Feasibility on the synthetic pairs.

Zero-shot CLIP on crops around the wrists, from the poses already extracted by the probe run. The
prompts and the crop are fixed here before any result. Five pairs show feasibility at most.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run

LEFT, RIGHT = (7, 9), (8, 10)  # (elbow, wrist), COCO-17


def wrist_boxes(keypoints: np.ndarray, box: np.ndarray, c: dict) -> list[tuple[int, int, int, int]]:
    """A square around each hand: the wrist pushed along the forearm, sized by the person."""
    side = c["crop_fraction"] * (box[3] - box[1])
    out = []
    for elbow, wrist in (LEFT, RIGHT):
        center = keypoints[wrist, :2] + c["forearm_push"] * (
            keypoints[wrist, :2] - keypoints[elbow, :2]
        )
        x, y = center - side / 2
        out.append((int(x), int(y), int(x + side), int(y + side)))
    return out


def crop(frame: np.ndarray, b: tuple[int, int, int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = max(b[0], 0), max(b[1], 0), min(b[2], width), min(b[3], height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((8, 8, 3), np.uint8)
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    run = start_run(
        "hand_object_probe", config=config, seeds={}, models={"clip": {"hub_id": config["model"]}}
    )
    model = CLIPModel.from_pretrained(config["model"]).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(config["model"])
    prompts = config["holding"] + config["empty"]
    n_hold = len(config["holding"])
    rows = []
    for pose in sorted(Path(config["poses"]).glob("p*_poses.npz")):
        vid = pose.name.removesuffix("_poses.npz")
        z = np.load(pose)
        capture = cv2.VideoCapture(str(Path(config["videos"]) / f"{vid}.mp4"))
        fps = capture.get(cv2.CAP_PROP_FPS)
        times, holds, sheet = [], [], []
        for i in range(0, len(z["frame_idx"]), config["every"]):
            t = z["frame_idx"][i] / config["pose_fps"]
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
            ok, frame = capture.read()
            if not ok:
                continue
            crops = [crop(frame, b) for b in wrist_boxes(z["keypoints"][i], z["bbox"][i], config)]
            inputs = processor(text=prompts, images=crops, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**inputs.to("cuda")).logits_per_image
            p = logits.softmax(-1)[:, :n_hold].sum(-1).cpu().numpy()
            times.append(t)
            holds.append(float(p.max()))
            best = crops[int(p.argmax())]
            sheet.append(cv2.resize(best, (96, 96))[:, :, ::-1])
        capture.release()
        cv2.imwrite(str(run / f"{vid}_hands.jpg"), np.hstack(sheet))
        tm, hold = np.asarray(times), np.asarray(holds)
        rows.append(
            {
                "video": vid,
                "start_0_3": float(np.median(hold[tm < 3])),
                "action_3_7": float(np.median(hold[(tm >= 3) & (tm < 7)])),
                "end_7_10": float(np.median(hold[tm >= 7])),
            }
        )
        print(rows[-1], flush=True)
    with (run / "hands.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("run", run)


if __name__ == "__main__":
    main()
