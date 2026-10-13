"""Video -> pose records in the ADR-0004 contract: YOLOv8 -> ByteTrack -> HRNet.

Runs in the separate pose environment (.venv-pose; session 2B). Ultralytics and MMPose
are imported inside `extract_video`, so this module can be imported without them.
Every stage is timed separately on the frames kept after resampling.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from tcc.data.schema import VideoPoses
from tcc.data.skeleton import V
from tcc.pipeline.resample import frames_to_keep

STAGES = ("decode", "detect", "track", "pose")


@dataclass
class Rows:
    """Pose rows accumulated frame by frame, converted to the contract at the end."""

    frame_idx: list[int] = field(default_factory=list)
    track_id: list[int] = field(default_factory=list)
    bbox: list[npt.NDArray[np.float32]] = field(default_factory=list)
    keypoints: list[npt.NDArray[np.float32]] = field(default_factory=list)

    def add(
        self,
        q: int,
        tracks: npt.NDArray[np.float32],
        xy: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
    ) -> None:
        """tracks: (k, >=5) rows [x1, y1, x2, y2, track_id, ...]; xy: (k, 17, 2); scores (k, 17)."""
        for i in range(len(tracks)):
            self.frame_idx.append(q)
            self.track_id.append(int(tracks[i, 4]))
            self.bbox.append(tracks[i, :4].astype(np.float32))
            self.keypoints.append(np.concatenate([xy[i], scores[i][:, None]], 1).astype(np.float32))

    def to_video(self, video_id: str, camera: str, n_frames: int) -> VideoPoses:
        n = len(self.frame_idx)
        return VideoPoses(
            video_id=video_id,
            camera=camera,
            n_frames=n_frames,
            frame_idx=np.asarray(self.frame_idx, np.int64),
            track_id=np.asarray(self.track_id, np.int64),
            bbox=np.stack(self.bbox) if n else np.zeros((0, 4), np.float32),
            keypoints=np.stack(self.keypoints) if n else np.zeros((0, V, 3), np.float32),
            frame_labels=None,
        )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def extract_video(
    path: Path, video_id: str, camera: str, cfg: dict[str, Any]
) -> tuple[VideoPoses, dict[str, Any]]:  # Any: YAML config and JSON-ready stats
    """Extract pose records from one video; return them and per-stage timings."""
    from mmengine.registry import init_default_scope
    from mmpose.apis import inference_topdown, init_model
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    det, trk, pose = cfg["detector"], cfg["tracker"], cfg["pose"]
    detector = YOLO(det["weights"])
    pose_model = init_model(pose["config"], pose["weights"], device=cfg["device"])
    init_default_scope("mmpose")

    cap = cv2.VideoCapture(str(path))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    n_source = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keep = frames_to_keep(source_fps, n_source, cfg["target_fps"])
    if cfg.get("max_output_frames"):
        keep = keep[: cfg["max_output_frames"]]
    keep_set = {int(i): k for k, i in enumerate(keep)}

    def detect(frame: Any) -> Any:  # Any: cv2 frame in, ultralytics Boxes out
        result = detector.predict(
            frame,
            classes=[0],
            conf=det["conf"],
            imgsz=det["imgsz"],
            device=cfg["device"],
            half=det["half"],
            verbose=False,
        )[0]
        return result.boxes.cpu().numpy()

    # warm-up on the first kept frame, outside the timings
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"cannot decode {path}")
    warm = detect(first)
    inference_topdown(pose_model, first, warm.xyxy[:1] if len(warm) else None, bbox_format="xyxy")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tracker = BYTETracker(IterableSimpleNamespace(**trk))
    times = {s: 0.0 for s in STAGES}
    detections_per_frame, tracks_per_frame = [], []
    rows = Rows()
    last = int(keep[-1]) if len(keep) else -1
    start_total = time.perf_counter()
    for i in range(last + 1):
        t0 = time.perf_counter()
        if not cap.grab():
            raise RuntimeError(f"{path}: decoding stopped at frame {i} of {n_source}")
        if i not in keep_set:
            times["decode"] += time.perf_counter() - t0
            continue
        ok, frame = cap.retrieve()
        times["decode"] += time.perf_counter() - t0
        if not ok:
            raise RuntimeError(f"{path}: cannot retrieve frame {i}")
        q = keep_set[i]

        t0 = time.perf_counter()
        boxes = detect(frame)
        _sync()
        times["detect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        tracks = tracker.update(boxes, frame)
        times["track"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if len(tracks):
            samples = inference_topdown(pose_model, frame, tracks[:, :4], bbox_format="xyxy")
            xy = np.concatenate([s.pred_instances.keypoints for s in samples])
            scores = np.concatenate([s.pred_instances.keypoint_scores for s in samples])
            _sync()
            rows.add(q, tracks, xy, scores)
        times["pose"] += time.perf_counter() - t0
        detections_per_frame.append(len(boxes))
        tracks_per_frame.append(len(tracks))
    total = time.perf_counter() - start_total
    cap.release()

    n_out = len(keep)
    stats = {
        "source_fps": source_fps,
        "source_frames": n_source,
        "source_size": [int(first.shape[1]), int(first.shape[0])],
        "output_frames": n_out,
        "seconds": {**times, "total": total},
        "fps": {s: (n_out / times[s] if times[s] else None) for s in STAGES},
        "fps_total": n_out / total,
        "persons_per_frame": {
            "detections_mean": float(np.mean(detections_per_frame)),
            "tracks_mean": float(np.mean(tracks_per_frame)),
            "tracks_max": int(np.max(tracks_per_frame)),
        },
        "vram_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "vram_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
    }
    return rows.to_video(video_id, camera, n_out), stats
