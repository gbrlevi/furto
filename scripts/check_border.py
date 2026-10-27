"""Session 3, task A2.2: do PoseLift training boxes fit the 1920 x 1080 image, per camera?

Training partition only. The rule (config `max_excess_fraction`) was fixed before
measuring; this script only applies it and records the outcome.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tcc.data.poselift import load_video
from tcc.data.split import official_split
from tcc.data.stats import box_excess
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/check_border.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = load_config(Path(config["dataset_config"]))
    config["dataset"] = dataset
    run_dir = start_run("check_border", config=config, seeds={})

    root = Path(dataset["root"])
    w, h = (int(v) for v in config["image_size"])
    limit = float(config["max_excess_fraction"])
    horizontal: dict[str, list[np.ndarray]] = defaultdict(list)
    vertical: dict[str, list[np.ndarray]] = defaultdict(list)
    videos: dict[str, int] = defaultdict(int)
    for vid in official_split(root)["train"]:
        video = load_video(root, "train", vid, dataset)
        dx, dy = box_excess(video.bbox, (w, h))
        horizontal[video.camera].append(dx)
        vertical[video.camera].append(dy)
        videos[video.camera] += 1

    cameras = {}
    for cam in sorted(horizontal):
        dx, dy = np.concatenate(horizontal[cam]), np.concatenate(vertical[cam])
        outside = (dx > 0) | (dy > 0)
        cameras[cam] = {
            "videos": videos[cam],
            "boxes": int(len(dx)),
            "boxes_outside": int(outside.sum()),
            "boxes_outside_fraction": float(outside.mean()),
            "max_excess_px": {"horizontal": float(dx.max()), "vertical": float(dy.max())},
            "max_excess_fraction": {
                "of_width": float(dx.max() / w),
                "of_height": float(dy.max() / h),
            },
            "within_rule": bool(dx.max() / w <= limit and dy.max() / h <= limit),
        }
    confirmed = all(c["within_rule"] for c in cameras.values())
    results = {
        "image_size": [w, h],
        "max_excess_fraction_rule": limit,
        "cameras": cameras,
        "resolution_confirmed": confirmed,
        "cameras_over_rule": [c for c, v in cameras.items() if not v["within_rule"]],
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1))
    print(run_dir)


if __name__ == "__main__":
    main()
