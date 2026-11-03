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


def onnx_estimators(config: dict, run_dir: Path) -> dict:
    """The HRNet of `config["pose"]` run by ONNX Runtime with the providers of
    `config["onnx"]` (TensorRT FP16 in sessions 3B and ADR-0016), named "pose"."""
    from mmpose.apis import init_model

    from tcc.pipeline.hrnet import add_cuda_dll_dirs, export_onnx, onnx_estimator

    o, pose = config["onnx"], config["pose"]
    model = init_model(pose["config"], pose["weights"], device=config["device"])
    model.test_cfg["flip_test"] = bool(pose["flip_test"])
    add_cuda_dll_dirs()
    path = run_dir / "hrnet.onnx"
    export_onnx(model, path, bool(o["fp16"]), int(o["opset"]))
    for p in o["providers"]:  # TensorRT engine cache outside the repository
        if "trt_engine_cache_path" in p.get("options", {}):
            Path(p["options"]["trt_engine_cache_path"]).mkdir(parents=True, exist_ok=True)
    providers = [(p["name"], p.get("options", {})) for p in o["providers"]]
    estimate, session = onnx_estimator(model, path, providers, bool(o["fp16"]))
    if o["providers"][0]["name"] not in session.get_providers():
        raise RuntimeError(f"{o['providers'][0]['name']} did not load: {session.get_providers()}")
    return {"pose": estimate}


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
    estimators = onnx_estimators(config, run_dir) if config.get("onnx") else None

    results = {}
    for video in config["videos"]:
        extracted, stats = extract_video(
            Path(video["path"]), video["id"], video["camera"], config, estimators
        )
        poses = extracted["pose"]
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
