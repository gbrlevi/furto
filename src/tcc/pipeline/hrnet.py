"""HRNet cost split and an ONNX Runtime version of its network (session 3B, tasks 3 and 4).

Runs in the separate pose environment (.venv-pose). `profiled_topdown` repeats the steps of
MMPose `inference_topdown` one by one (same pipeline, same batch, same head and decoder) and
times each; its output must equal `inference_topdown`. `onnx_estimator` keeps the MMPose
preprocessing and decoding and replaces only the network forward (backbone and heatmap
head) by an ONNX Runtime session, so any pose difference comes from the network alone.
"""

import importlib.util
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from tcc.pipeline.pose_extract import FloatArray, PoseEstimator

PROFILE_STAGES = ("preprocess_cpu", "to_gpu_normalize", "backbone", "head", "decode_cpu")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _batch(model: Any, frame: npt.NDArray[np.uint8], boxes: FloatArray) -> Any:
    """MMPose test pipeline on every box, collated into one batch (as `inference_topdown`)."""
    from mmcv.transforms import Compose
    from mmengine.dataset import pseudo_collate

    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    data = []
    for box in boxes:
        info = {"img": frame, "bbox": box[None], "bbox_score": np.ones(1, np.float32)}
        info.update(model.dataset_meta)
        data.append(pipeline(info))
    return pseudo_collate(data)


def _heatmaps_from(model: Any, feats: Any, flip_feats: Any, samples: Any) -> torch.Tensor:
    """Head forward, averaged with the flipped pass when there is one (HeatmapHead.predict)."""
    from mmpose.models.utils.tta import flip_heatmaps

    head, cfg = model.head, model.test_cfg
    heatmaps: torch.Tensor = head.forward(feats)
    if flip_feats is not None:
        flipped = flip_heatmaps(
            head.forward(flip_feats),
            flip_mode=cfg.get("flip_mode", "heatmap"),
            flip_indices=samples[0].metainfo["flip_indices"],
            shift_heatmap=cfg.get("shift_heatmap", False),
        )
        heatmaps = (heatmaps + flipped) * 0.5
    return heatmaps


def _decode(model: Any, heatmaps: torch.Tensor, samples: Any) -> tuple[FloatArray, FloatArray]:
    preds = model.head.decode(heatmaps)
    results = model.add_pred_to_datasample(preds, None, samples)
    xy = np.concatenate([r.pred_instances.keypoints for r in results])
    scores = np.concatenate([r.pred_instances.keypoint_scores for r in results])
    return xy.astype(np.float32), scores.astype(np.float32)


@torch.no_grad()
def profiled_topdown(
    model: Any, frame: npt.NDArray[np.uint8], boxes: FloatArray
) -> tuple[FloatArray, FloatArray, dict[str, float]]:  # Any: MMPose model
    """`inference_topdown` for all boxes of a frame in one batch, timed per stage (seconds,
    with a CUDA synchronization at the end of each GPU stage)."""
    times = {}
    t0 = time.perf_counter()
    batch = _batch(model, frame, boxes)
    times["preprocess_cpu"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    data = model.data_preprocessor(batch, False)
    _sync()
    times["to_gpu_normalize"] = time.perf_counter() - t0

    inputs, samples = data["inputs"], data["data_samples"]
    flip = model.test_cfg.get("flip_test", False)
    t0 = time.perf_counter()
    feats = model.extract_feat(inputs)
    flip_feats = model.extract_feat(inputs.flip(-1)) if flip else None
    _sync()
    times["backbone"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    heatmaps = _heatmaps_from(model, feats, flip_feats, samples)
    _sync()
    times["head"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    xy, scores = _decode(model, heatmaps, samples)
    times["decode_cpu"] = time.perf_counter() - t0
    return xy, scores, times


class _Network(torch.nn.Module):
    """Backbone and heatmap head of an MMPose top-down model: image tensor -> heatmaps."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.backbone, self.neck, self.head = model.backbone, model.neck, model.head
        self.with_neck = model.with_neck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.with_neck:
            feats = self.neck(feats)
        out: torch.Tensor = self.head.forward(feats)
        return out


@torch.no_grad()
def export_onnx(model: Any, path: Path, fp16: bool, opset: int) -> None:
    """Export backbone + head with a dynamic batch. FP16 exports the half-precision network
    (on the GPU), so the ONNX graph takes and returns float16."""
    net = _Network(model).eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, 256, 192, device=device)
    if fp16:
        net, x = net.half(), x.half()
    torch.onnx.export(
        net, (x,), str(path), input_names=["input"], output_names=["heatmaps"],
        dynamic_axes={"input": {0: "batch"}, "heatmaps": {0: "batch"}}, opset_version=opset,
    )  # fmt: skip
    if fp16:
        net.float()


def add_cuda_dll_dirs() -> list[str]:
    """Put the CUDA and cuDNN DLLs of torch and, when installed, the TensorRT DLLs
    (`tensorrt_libs`) on the DLL search path, so ONNX Runtime providers can load them on
    Windows. Returns the directories added."""
    dirs = [str(Path(torch.__file__).parent / "lib")]
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is not None and spec.submodule_search_locations:
        dirs.append(str(next(iter(spec.submodule_search_locations))))
    for d in dirs:
        os.add_dll_directory(d)
    os.environ["PATH"] = os.pathsep.join([*dirs, os.environ.get("PATH", "")])
    return dirs


def onnx_estimator(
    model: Any, onnx_path: Path, providers: list[Any], fp16: bool
) -> tuple[PoseEstimator, Any]:  # Any: MMPose model and ONNX Runtime session
    """Pose estimator with the MMPose pipeline and decoder around an ONNX Runtime network.
    Test-time flip follows `model.test_cfg`; the flipped crops go in the same batch."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    dtype = np.float16 if fp16 else np.float32
    flip = model.test_cfg.get("flip_test", False)

    @torch.no_grad()
    def estimate(frame: npt.NDArray[np.uint8], boxes: FloatArray) -> tuple[FloatArray, FloatArray]:
        batch = _batch(model, frame, boxes)
        data = model.data_preprocessor(batch, False)
        inputs, samples = data["inputs"], data["data_samples"]
        x = torch.cat([inputs, inputs.flip(-1)]) if flip else inputs
        out = session.run(["heatmaps"], {"input": x.cpu().numpy().astype(dtype)})[0]
        heat = torch.from_numpy(out.astype(np.float32)).to(inputs.device)
        n = len(inputs)
        if flip:
            from mmpose.models.utils.tta import flip_heatmaps

            cfg = model.test_cfg
            flipped = flip_heatmaps(
                heat[n:], flip_mode=cfg.get("flip_mode", "heatmap"),
                flip_indices=samples[0].metainfo["flip_indices"],
                shift_heatmap=cfg.get("shift_heatmap", False),
            )  # fmt: skip
            heat = (heat[:n] + flipped) * 0.5
        return _decode(model, heat, samples)

    return estimate, session
