"""Render one PoseLift video as skeletons on a neutral background, as mp4 in runs/."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from tcc.data.poselift import load_video
from tcc.data.split import official_split
from tcc.utils.config import load_config
from tcc.utils.render import draw_skeletons
from tcc.utils.run_logger import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/render_skeleton.yaml"))
    parser.add_argument("--video-id", help="overrides video_id from the config")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = load_config(Path(config["dataset_config"]))
    config["dataset"] = dataset
    if args.video_id:
        config["video_id"] = args.video_id

    root = Path(dataset["root"])
    split = config["split"]
    video_id = config["video_id"] or official_split(root)[split][0]
    config["video_id"] = video_id
    run_dir = start_run("render_skeleton", config=config, seeds={})

    video = load_video(root, split, video_id, dataset)
    width, height = config["image_size"]
    scale = config["scale"]
    out = run_dir / f"{video_id}.mp4"
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter.fourcc(*"mp4v"),
        float(dataset["fps"]),
        (int(width * scale), int(height * scale)),
    )
    for q in range(video.n_frames):
        rows = np.flatnonzero(video.frame_idx == q)
        caption = f"{video_id} [{split}] frame {q}"
        frame = draw_skeletons(
            video.keypoints[rows],
            video.track_id[rows],
            (width, height),
            scale,
            config["conf_min"],
            caption,
        )
        writer.write(frame)
    writer.release()
    print(out)


if __name__ == "__main__":
    main()
