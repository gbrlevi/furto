"""Person boxes every few frames of each video, for actor-centred clips. Run with .venv-pose."""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    p = config["people"]
    detector = YOLO(p["weights"])
    out = Path(p["out"])
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        f
        for d in config["videos"]
        for f in sorted(Path(d).iterdir())
        if f.suffix in (".mp4", ".avi")
    ]
    for i, path in enumerate(paths):
        target = out / f"{path.stem}.npz"
        if target.exists():
            continue
        capture = cv2.VideoCapture(str(path))
        rows: list[list[float]] = []
        n = 0
        while capture.grab():
            if n % p["every"] == 0:
                _, frame = capture.retrieve()
                result = detector.predict(
                    frame, classes=[0], conf=p["conf"], imgsz=p["imgsz"], verbose=False
                )[0]
                rows += [[n, *b] for b in result.boxes.xyxy.cpu().numpy().tolist()]
            n += 1
        capture.release()
        np.savez_compressed(target, boxes=np.asarray(rows, np.float32).reshape(-1, 5), n_frames=n)
        print(i + 1, "of", len(paths), path.stem, len(rows), flush=True)


if __name__ == "__main__":
    main()
