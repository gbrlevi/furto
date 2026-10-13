"""Extract pose records from videos with YOLOv8 + ByteTrack + HRNet (session 2B).

Run with the pose environment:

    .venv-pose/Scripts/python.exe scripts/extract_pose.py --config configs/pose_extract.yaml

Writes, per video, <name>_poses.npz (ADR-0004 contract) and the stage timings to
results.json; with `copy_to`, also a copy of each npz in that folder. Raw video never
leaves data/.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from tcc.pipeline.pose_extract import extract_video
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pose_extract.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    det, pose = config["detector"], config["pose"]
    version = importlib.metadata.version
    models = {
        "detector": {
            "name": det["name"],
            "version": f"ultralytics {version('ultralytics')}",
            "weights": det["weights"],
        },
        "tracker": {
            "name": "ByteTrack (ultralytics.trackers.byte_tracker.BYTETracker)",
            "version": f"ultralytics {version('ultralytics')}",
        },
        "pose": {
            "name": pose["name"],
            "version": f"mmpose {version('mmpose')}, mmcv {version('mmcv')}",
            "weights": pose["weights"],
        },
    }
    run_dir = start_run(config["name"], config=config, seeds={}, models=models)

    results = {}
    for video in config["videos"]:
        poses, stats = extract_video(Path(video["path"]), video["id"], video["camera"], config)
        arrays = {
            "frame_idx": poses.frame_idx,
            "track_id": poses.track_id,
            "bbox": poses.bbox,
            "keypoints": poses.keypoints,
            "n_frames": poses.n_frames,
        }
        np.savez_compressed(run_dir / f"{video['id']}_poses.npz", **arrays)
        if config.get("copy_to"):  # e.g. data/interim/demo (session 2C)
            copy_dir = Path(config["copy_to"])
            copy_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(copy_dir / f"{video['id']}_poses.npz", **arrays)
        stats["records"] = len(poses)
        stats["tracks"] = int(len(np.unique(poses.track_id)))
        results[video["id"]] = stats
        print(
            video["id"], json.dumps(stats["fps"]), f"total {stats['fps_total']:.2f} fps", flush=True
        )
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
