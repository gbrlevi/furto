"""Render the skeletons of an extraction run (<id>_poses.npz) on a neutral background.

Only skeletons are drawn; no pixel of the source video is read (ADR-0005).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tcc.utils.config import load_config
from tcc.utils.render import draw_skeletons
from tcc.utils.run_logger import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = start_run("render_extracted", config=config, seeds={})

    source = Path(config["extraction_run"])
    results = json.loads((source / "results.json").read_text(encoding="utf-8"))
    for video_id in config["videos"]:
        if not isinstance(video_id, str):
            raise SystemExit(f"video ids must be quoted strings, got {video_id!r}")
        z = np.load(source / f"{video_id}_poses.npz")
        width, height = results[video_id]["source_size"]
        scale = config["scale"]
        writer = cv2.VideoWriter(
            str(run_dir / f"{video_id}_skeleton.mp4"),
            cv2.VideoWriter.fourcc(*"mp4v"),
            float(config["fps"]),
            (int(width * scale), int(height * scale)),
        )
        for q in range(int(z["n_frames"])):
            rows = np.flatnonzero(z["frame_idx"] == q)
            writer.write(
                draw_skeletons(
                    z["keypoints"][rows],
                    z["track_id"][rows],
                    (width, height),
                    scale,
                    config["conf_min"],
                    f"{video_id} frame {q}",
                )
            )
        writer.release()
    print(run_dir)


if __name__ == "__main__":
    main()
