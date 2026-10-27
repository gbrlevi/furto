"""Online pose process (session 3B, task 6): video file -> contract records on stdout.

Started by scripts/online_score.py in the pose environment, with this checkout's sources
first on the path. Reads the video at the target fps, never decoding frame q before q / fps
seconds after the start (the pace of the video; frames are never skipped, P14 item 4),
runs YOLOv8 -> ByteTrack -> HRNet (ADR-0012, ADR-0013) and writes one JSON line per output
frame, also when nobody is there. Channel: the stdout pipe of the child process (one
producer and one consumer on the same machine, no extra dependency, back-pressure from the
pipe buffer). Library output is sent to stderr so stdout carries only the records.
"""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    channel = open(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)  # noqa: SIM115
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # C-level prints of the libraries too
    sys.stdout = sys.stderr

    from tcc.pipeline.pose_extract import VideoInfo, frame_stream, mmpose_estimator
    from tcc.utils.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    pose = cfg["pose"]
    estimator = mmpose_estimator(pose, cfg["device"], pose.get("flip_test"))
    times: dict[str, float] = {}
    info = VideoInfo()
    first = True
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "ultralytics", "mmpose", "mmcv", "mmengine", "numpy"]
    }
    for fr in frame_stream(args.video, cfg, {"pose": estimator}, times, info, cfg["target_fps"]):
        if first:
            header = {
                "type": "header", "source_fps": info.source_fps,
                "source_frames": info.source_frames, "source_size": list(info.source_size),
                "output_frames": info.output_frames, "fps": cfg["target_fps"],
                "pose_env_versions": versions, "pid": os.getpid(),
            }  # fmt: skip
            channel.write(json.dumps(header) + "\n")
            first = False
        xy, scores = fr.poses.get("pose", (np.zeros((0, 17, 2)), np.zeros((0, 17))))
        kp = np.concatenate([xy, scores[..., None]], -1) if len(fr.tracks) else np.zeros((0, 17, 3))
        message = {
            "type": "frame",
            "q": fr.q,
            "read_time": fr.read_time,
            "track_id": [int(v) for v in fr.tracks[:, 4]] if len(fr.tracks) else [],
            "bbox": np.asarray(fr.tracks[:, :4], np.float32).tolist() if len(fr.tracks) else [],
            "keypoints": np.asarray(kp, np.float32).tolist(),
        }
        channel.write(json.dumps(message) + "\n")
    channel.write(json.dumps({"type": "end", "seconds": times}) + "\n")
    channel.close()


if __name__ == "__main__":
    main()
