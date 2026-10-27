"""Session 3B, task 2: HRNet test-time flip on vs off on the same tracked boxes.

Run with the pose environment and this checkout's sources first on the path:

    PYTHONPATH=src .venv-pose/Scripts/python.exe scripts/measure_flip.py

Writes, per clip, <id>_flip_on.npz and <id>_flip_off.npz (ADR-0004 contract, skeletons
only) and results.json with the displacement statistics and the outcome of the rule fixed
in the config. Raw video never leaves data/.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from tcc.data.schema import VideoPoses
from tcc.pipeline.pose_compare import displacement_summary, normalized_displacement
from tcc.pipeline.pose_extract import extract_video, mmpose_estimator
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def _save(path: Path, poses: VideoPoses) -> None:
    np.savez_compressed(
        path, frame_idx=poses.frame_idx, track_id=poses.track_id, bbox=poses.bbox,
        keypoints=poses.keypoints, n_frames=poses.n_frames,
    )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/measure_flip.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    c1 = load_config(Path(config["c1_config"]))
    noise = float(c1["augmentation"]["noise"])
    conf_min = float(c1["treatment"]["conf_min"])
    config["rule_inputs"] = {"augmentation_noise": noise, "conf_min": conf_min}
    det, pose = config["detector"], config["pose"]
    version = importlib.metadata.version
    models = {
        "detector": {"name": det["name"], "version": f"ultralytics {version('ultralytics')}",
                     "weights": det["weights"]},
        "tracker": {"name": "ByteTrack (ultralytics BYTETracker)",
                    "version": f"ultralytics {version('ultralytics')}"},
        "pose": {"name": pose["name"], "version": f"mmpose {version('mmpose')}",
                 "weights": pose["weights"]},
    }  # fmt: skip
    run_dir = start_run(config["name"], config=config, seeds={}, models=models)

    estimators = {
        "flip_on": mmpose_estimator(pose, config["device"], flip_test=True),
        "flip_off": mmpose_estimator(pose, config["device"], flip_test=False),
    }
    results, pooled = {}, []
    for video in config["videos"]:
        vid = video["id"]
        extracted, stats = extract_video(
            Path(video["path"]), vid, video["camera"], config, estimators
        )
        on, off = extracted["flip_on"], extracted["flip_off"]
        _save(run_dir / f"{vid}_flip_on.npz", on)
        _save(run_dir / f"{vid}_flip_off.npz", off)
        d = normalized_displacement(on.keypoints, off.keypoints, conf_min)
        pooled.append(d)
        results[vid] = {"displacement": displacement_summary(d), "extraction": stats}
        print(vid, json.dumps(results[vid]["displacement"]), flush=True)
    summary = displacement_summary(np.concatenate(pooled))
    flip_stays = not (summary["mean"] is not None and summary["mean"] < noise)
    results["pooled"] = {
        "displacement": summary,
        "augmentation_noise": noise,
        "flip_in_operation": flip_stays,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["pooled"], indent=1))
    print(run_dir)


if __name__ == "__main__":
    main()
