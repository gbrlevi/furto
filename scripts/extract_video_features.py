"""Pretrained clip features of each video, one .npz per video."""

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification

from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run
from tcc.video.clips import clip_starts

MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet statistics the model expects
STD = np.array([0.229, 0.224, 0.225], np.float32)


@torch.no_grad()
def embed(model: VideoMAEForVideoClassification, clips: list[np.ndarray]) -> np.ndarray:
    x = (np.stack(clips).astype(np.float32) / 255.0 - MEAN) / STD  # B T H W C
    t = torch.from_numpy(x).permute(0, 1, 4, 2, 3).to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = model.videomae(pixel_values=t).last_hidden_state
        pooled = model.fc_norm(hidden.mean(1))
    return np.asarray(pooled.float().cpu().numpy(), np.float32)


def extract(model: VideoMAEForVideoClassification, path: Path, out: Path, c: dict) -> None:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    span = (c["num_frames"] - 1) * c["frame_step"] + 1
    recent: deque[np.ndarray] = deque(maxlen=span)
    feats, starts, batch = [], [], []
    n = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        recent.append(cv2.cvtColor(cv2.resize(frame, (c["size"], c["size"])), cv2.COLOR_BGR2RGB))
        n += 1
        if n >= span and (n - span) % c["hop"] == 0:
            frames = list(recent)
            batch.append(np.stack(frames[:: c["frame_step"]]))
            starts.append(n - span)
            if len(batch) == c["batch"]:
                feats.append(embed(model, batch))
                batch = []
    capture.release()
    if batch:
        feats.append(embed(model, batch))
    assert starts == clip_starts(n, span, c["hop"]).tolist()
    x = np.concatenate(feats) if feats else np.zeros((0, 768), np.float32)
    np.savez_compressed(
        out, x=x, starts=np.asarray(starts, np.int64), n_frames=n, fps=fps, span=span
    )


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    run = start_run(
        "extract_video_features",
        config=config,
        seeds={},
        models={"videomae": {"hub_id": config["model"]}},
    )
    model = VideoMAEForVideoClassification.from_pretrained(config["model"]).to("cuda").eval()
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        p
        for d in config["videos"]
        for p in sorted(Path(d).iterdir())
        if p.suffix in (".mp4", ".avi")
    ]
    for i, path in enumerate(paths):
        target = out / f"{path.stem}.npz"
        if not target.exists():
            extract(model, path, target, config)
            print(i + 1, "of", len(paths), path.stem, flush=True)
    print("run", run)


if __name__ == "__main__":
    main()
