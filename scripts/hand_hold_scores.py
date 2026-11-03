"""CLIP features of both hands of every pose record, and the zero-shot probability that the
hand holds a product.

Usage: hand_hold_scores.py CONFIG. For each video, reads its poses (15 fps, ADR-0004), decodes
the source frames those poses came from, crops a square around each hand (the wrist pushed
along the forearm, sized by the person box) and writes <out>/<id>.npz with `emb` (N, 2, D)
float16 image embeddings, `hold` (N, 2) zero-shot probabilities and the pose arrays. Videos
already written are skipped.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from tcc.pipeline.resample import frames_to_keep
from tcc.rules.shopping import hand_points
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def boxes(kp: np.ndarray, box: np.ndarray, c: dict) -> np.ndarray:
    """(N, 2, 4) square crops around the hands, side = fraction of the larger box side."""
    side = np.maximum(c["crop_fraction"] * np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]),
                      c["min_crop"])  # fmt: skip
    centre = hand_points(kp, c["forearm_push"])
    half = side[:, None, None] / 2
    return np.concatenate([centre - half, centre + half], axis=2)


def crop(frame: np.ndarray, b: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = max(int(b[0]), 0), max(int(b[1]), 0)
    x2, y2 = min(int(b[2]), w), min(int(b[3]), h)
    if x2 - x1 < 8 or y2 - y1 < 8:  # also keeps the channel axis unambiguous
        return np.zeros((8, 8, 3), np.uint8)
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def encode(
    model: CLIPModel,
    processor: CLIPProcessor,
    text: torch.Tensor,
    scale: torch.Tensor,
    n_hold: int,
    images: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized image embeddings and the summed probability of the holding prompts."""
    inputs = processor(images=images, return_tensors="pt").to("cuda")
    with torch.no_grad():
        e = model.get_image_features(**inputs).pooler_output
        e = e / e.norm(dim=-1, keepdim=True)
        p = (scale * e @ text.T).softmax(-1)[:, :n_hold].sum(-1)
    return e.cpu().numpy(), p.cpu().numpy()


def fill(
    emb: np.ndarray, hold: np.ndarray, slots: list[tuple[int, int]], e: np.ndarray, p: np.ndarray
) -> None:
    for (row, hand), ei, pi in zip(slots, e, p, strict=True):
        emb[row, hand], hold[row, hand] = ei, pi


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    run = start_run(
        "hand_hold_scores", config=config, seeds={}, models={"clip": {"hub_id": config["model"]}}
    )
    model = CLIPModel.from_pretrained(config["model"]).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(config["model"])
    prompts = config["holding"] + config["empty"]
    with torch.no_grad():
        text = processor(text=prompts, return_tensors="pt", padding=True).to("cuda")
        t = model.get_text_features(**text).pooler_output
        t = t / t.norm(dim=-1, keepdim=True)
    scale = model.logit_scale.exp()
    n_hold = len(config["holding"])
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    poses = sorted(Path(config["poses"]).glob("*_poses.npz"))
    for n, path in enumerate(poses, 1):
        vid = path.name.removesuffix("_poses.npz")
        target = out / f"{vid}.npz"
        if target.exists():
            continue
        z = np.load(path)
        kp, box, fidx = z["keypoints"], z["bbox"], z["frame_idx"]
        cuts = boxes(kp, box, config)
        name = config["video_name"].format(vid)
        dirs = config["videos"] if isinstance(config["videos"], list) else [config["videos"]]
        capture = cv2.VideoCapture(
            str(next(Path(d) / name for d in dirs if (Path(d) / name).exists()))
        )
        fps, total = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source = frames_to_keep(fps, total, config["pose_fps"])
        wanted = {int(source[q]): q for q in np.unique(fidx) if q < len(source)}
        emb = np.zeros((len(fidx), 2, model.config.projection_dim), np.float16)
        hold = np.full((len(fidx), 2), np.nan, np.float32)
        images: list[np.ndarray] = []
        slots: list[tuple[int, int]] = []

        i = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            q = wanted.get(i)
            i += 1
            if q is None:
                continue
            for row in np.flatnonzero(fidx == q):
                for hand in (0, 1):
                    images.append(crop(frame, cuts[row, hand]))
                    slots.append((int(row), hand))
            if len(images) >= config["batch"]:
                fill(emb, hold, slots, *encode(model, processor, t, scale, n_hold, images))
                images, slots = [], []
        if images:
            fill(emb, hold, slots, *encode(model, processor, t, scale, n_hold, images))
        capture.release()
        np.savez_compressed(
            target, emb=emb, hold=hold, frame_idx=fidx, track_id=z["track_id"], bbox=box,
            keypoints=kp, n_frames=z["n_frames"],
        )  # fmt: skip
        print(f"{n} of {len(poses)} {vid} rows {len(fidx)}", flush=True)
    print("run", run)


if __name__ == "__main__":
    main()
