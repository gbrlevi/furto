"""Session 3B, tasks 3 and 4: HRNet cost per stage and ONNX Runtime versions of the network.

Run with the pose environment and this checkout's sources first on the path:

    PYTHONPATH=src .venv-pose/Scripts/python.exe scripts/bench_pose_opt.py

Every variant runs inside the same frame loop on the same tracked boxes: the MMPose
reference (`inference_topdown`), the same steps timed one by one, and ONNX Runtime
networks (CUDA FP32, CUDA FP16 and, when the TensorRT provider loads, TensorRT FP16).
Poses of each variant are compared with the reference by the normalized displacement of
task 2 (limit: the augmentation noise). Only measures; the extraction config is not
changed here.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcc.pipeline.hrnet import (
    PROFILE_STAGES,
    add_cuda_dll_dirs,
    export_onnx,
    onnx_estimator,
    profiled_topdown,
)
from tcc.pipeline.pose_compare import displacement_summary, normalized_displacement
from tcc.pipeline.pose_extract import (
    STAGES,
    FloatArray,
    PoseEstimator,
    VideoInfo,
    frame_stream,
)
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def _counted(model: Any, calls: list[int]) -> PoseEstimator:
    """MMPose reference that counts the batched network calls (`test_step`) per frame."""
    from mmpose.apis import inference_topdown

    original = model.test_step

    def counting(data: Any) -> Any:
        calls[-1] += 1
        return original(data)

    model.test_step = counting

    def estimate(frame: Any, boxes: FloatArray) -> tuple[FloatArray, FloatArray]:
        calls.append(0)
        samples = inference_topdown(model, frame, boxes, bbox_format="xyxy")
        xy = np.concatenate([s.pred_instances.keypoints for s in samples])
        scores = np.concatenate([s.pred_instances.keypoint_scores for s in samples])
        return xy.astype(np.float32), scores.astype(np.float32)

    return estimate


def _profiled(model: Any, stage_times: dict[str, float], batch_sizes: list[int]) -> PoseEstimator:
    def estimate(frame: Any, boxes: FloatArray) -> tuple[FloatArray, FloatArray]:
        xy, scores, t = profiled_topdown(model, frame, boxes)
        for k, v in t.items():
            stage_times[k] += v
        batch_sizes.append(len(boxes))
        return xy, scores

    return estimate


def main() -> None:
    from mmpose.apis import init_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bench_pose_opt.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    c1 = load_config(Path(config["c1_config"]))
    noise, conf_min = float(c1["augmentation"]["noise"]), float(c1["treatment"]["conf_min"])
    import onnxruntime as ort

    version = importlib.metadata.version
    pose = config["pose"]
    models = {
        "detector": {"name": config["detector"]["name"],
                     "version": f"ultralytics {version('ultralytics')}",
                     "weights": config["detector"]["weights"]},
        "pose": {"name": pose["name"], "version": f"mmpose {version('mmpose')}",
                 "weights": pose["weights"]},
        "onnxruntime": {"name": "onnxruntime-gpu", "version": ort.__version__},
    }  # fmt: skip
    run_dir = start_run(config["name"], config=config, seeds={}, models=models)

    model = init_model(pose["config"], pose["weights"], device=config["device"])
    model.test_cfg["flip_test"] = bool(config["flip_test"])
    calls: list[int] = []
    stage_times = dict.fromkeys(PROFILE_STAGES, 0.0)
    batch_sizes: list[int] = []
    estimators: dict[str, PoseEstimator] = {
        "mmpose": _counted(model, calls),
        "mmpose_profiled": _profiled(model, stage_times, batch_sizes),
    }
    sessions: dict[str, Any] = {}
    results: dict[str, Any] = {"dll_dirs": add_cuda_dll_dirs(), "videos": {}}
    providers_used: dict[str, list[str]] = {}
    for name, v in config["onnx_variants"].items():
        path = run_dir / f"hrnet_w32_{'fp16' if v['fp16'] else 'fp32'}.onnx"
        for p in v["providers"]:  # TensorRT engine cache outside the repository
            if "trt_engine_cache_path" in p.get("options", {}):
                Path(p["options"]["trt_engine_cache_path"]).mkdir(parents=True, exist_ok=True)
        if not path.exists():
            export_onnx(model, path, bool(v["fp16"]), int(config["opset"]))
        providers = [(p["name"], p.get("options", {})) for p in v["providers"]]
        try:
            estimators[name], sessions[name] = onnx_estimator(model, path, providers, v["fp16"])
            providers_used[name] = sessions[name].get_providers()
        except Exception as error:  # noqa: BLE001 - the provider may not load on Windows
            providers_used[name] = [f"failed: {type(error).__name__}: {error}"]
            continue
        if v["providers"][0]["name"] not in providers_used[name]:
            del estimators[name]  # fell back to another provider: not the variant asked for
    print(json.dumps(providers_used, indent=1), flush=True)

    results["providers"] = providers_used
    for video in config["videos"]:
        times: dict[str, float] = {}
        info = VideoInfo()
        poses: dict[str, list[np.ndarray]] = {k: [] for k in estimators}
        tracked = []
        calls.clear()
        for k in stage_times:
            stage_times[k] = 0.0
        batch_sizes.clear()
        for fr in frame_stream(Path(video["path"]), config, estimators, times, info):
            tracked.append(len(fr.tracks))
            for name, (xy, sc) in fr.poses.items():
                poses[name].append(np.concatenate([xy, sc[..., None]], -1))
        n = info.output_frames
        with_people = int(np.count_nonzero(tracked))
        base = sum(times[s] for s in STAGES)
        ref = np.concatenate(poses["mmpose"]) if poses["mmpose"] else np.zeros((0, 17, 3))
        per_variant = {}
        for name in estimators:
            kp = np.concatenate(poses[name]) if poses[name] else np.zeros((0, 17, 3))
            d = displacement_summary(normalized_displacement(ref, kp, conf_min))
            per_variant[name] = {
                "pose_seconds": times[name],
                "ms_per_frame_with_people": 1000 * times[name] / max(with_people, 1),
                "stack_fps": n / (base + times[name]),
                "displacement_vs_mmpose": d,
                "within_noise": d["mean"] is not None and d["mean"] < noise,
                "max_abs_difference_px": float(np.abs(kp[..., :2] - ref[..., :2]).max())
                if len(ref) else None,
                "max_abs_difference_confidence": float(np.abs(kp[..., 2] - ref[..., 2]).max())
                if len(ref) else None,
            }  # fmt: skip
        results["videos"][video["id"]] = {
            "output_frames": n,
            "frames_with_people": with_people,
            "persons_per_frame": float(np.mean(tracked)),
            "persons_per_frame_with_people": float(np.mean([t for t in tracked if t])),
            "network_calls_per_frame_with_people": (
                {"mean": float(np.mean(calls)), "max": int(np.max(calls))} if calls else None
            ),
            "stage_seconds": {s: times[s] for s in STAGES},
            "profile_ms_per_frame_with_people": {
                k: 1000 * v / max(with_people, 1) for k, v in stage_times.items()
            },
            "profile_batch_size_mean": float(np.mean(batch_sizes)) if batch_sizes else None,
            "variants": per_variant,
        }
        print(video["id"], json.dumps(results["videos"][video["id"]], indent=1), flush=True)
    results["augmentation_noise"] = noise
    results["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for p in run_dir.glob("*.onnx"):
        p.unlink()  # derived weights: re-exportable, not kept in runs/
    print(run_dir)


if __name__ == "__main__":
    main()
