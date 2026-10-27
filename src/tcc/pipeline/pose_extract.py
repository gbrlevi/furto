"""Video -> pose records in the ADR-0004 contract: YOLOv8 -> ByteTrack -> HRNet.

Runs in the separate pose environment (.venv-pose; session 2B). Ultralytics and MMPose
are imported inside the functions that need them, so this module can be imported
without them. Every stage is timed separately on the frames kept after resampling.

`frame_stream` is the one frame loop: the offline extraction (`extract_video`) and the
online pose process (session 3B) both consume it. Several pose estimators can run on the
same tracked boxes (session 3B: test-time flip on and off, MMPose against ONNX).
"""

import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import numpy.typing as npt
import torch

from tcc.data.schema import VideoPoses
from tcc.data.skeleton import V
from tcc.pipeline.resample import frames_to_keep

STAGES = ("decode", "detect", "track")

FloatArray = npt.NDArray[np.float32]
# frame (BGR), boxes (k, 4) x1 y1 x2 y2 -> keypoints xy (k, 17, 2) and scores (k, 17)
PoseEstimator = Callable[[npt.NDArray[np.uint8], FloatArray], tuple[FloatArray, FloatArray]]


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


def mmpose_estimator(
    pose: Mapping[str, Any], device: str, flip_test: bool | None = None
) -> PoseEstimator:  # Any: the `pose` section of the YAML config
    """HRNet through `inference_topdown`: all boxes of a frame go in one batch. `flip_test`
    None keeps the value of the MMPose config file (True for the COCO HRNet configs)."""
    from mmpose.apis import inference_topdown, init_model

    model = init_model(pose["config"], pose["weights"], device=device)
    if flip_test is not None:
        model.test_cfg["flip_test"] = flip_test

    def estimate(frame: npt.NDArray[np.uint8], boxes: FloatArray) -> tuple[FloatArray, FloatArray]:
        samples = inference_topdown(model, frame, boxes, bbox_format="xyxy")
        xy = np.concatenate([s.pred_instances.keypoints for s in samples])
        scores = np.concatenate([s.pred_instances.keypoint_scores for s in samples])
        return xy.astype(np.float32), scores.astype(np.float32)

    return estimate


class Frame(NamedTuple):
    q: int  # output frame index at the target fps
    read_time: float  # time.time() right after decoding (for end-to-end latency)
    detections: int  # person boxes from the detector
    tracks: FloatArray  # (k, >=5) [x1, y1, x2, y2, track_id, ...]
    poses: dict[str, tuple[FloatArray, FloatArray]]  # estimator -> (xy, scores)


@dataclass
class VideoInfo:
    source_fps: float = 0.0
    source_frames: int = 0
    source_size: tuple[int, int] = (0, 0)
    output_frames: int = 0


def frame_stream(
    path: Path,
    cfg: Mapping[str, Any],  # Any: YAML config (detector, tracker, device, target_fps)
    estimators: Mapping[str, PoseEstimator],
    times: dict[str, float],
    info: VideoInfo,
    pace_fps: float | None = None,
) -> Iterator[Frame]:
    """Decode at `target_fps`, detect, track and run every estimator on the tracked boxes,
    one kept frame at a time. Stage times are added to `times` (decode, detect, track and
    one entry per estimator). With `pace_fps`, frame q is not decoded before q / pace_fps
    seconds after the start (frames emitted at the pace of the video, never skipped)."""
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    det = cfg["detector"]
    detector = YOLO(det["weights"])
    cap = cv2.VideoCapture(str(path))
    info.source_fps = cap.get(cv2.CAP_PROP_FPS)
    info.source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keep = frames_to_keep(info.source_fps, info.source_frames, cfg["target_fps"])
    if cfg.get("max_output_frames"):
        keep = keep[: cfg["max_output_frames"]]
    info.output_frames = len(keep)
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

    # warm-up on the first frame, outside the timings
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"cannot decode {path}")
    info.source_size = (int(first.shape[1]), int(first.shape[0]))
    warm = detect(first)
    warm_boxes = (
        warm.xyxy[:1].astype(np.float32)
        if len(warm)
        else np.array([[0, 0, first.shape[1], first.shape[0]]], np.float32)
    )
    for estimate in estimators.values():
        estimate(np.asarray(first, np.uint8), warm_boxes)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tracker = BYTETracker(IterableSimpleNamespace(**cfg["tracker"]))
    for s in (*STAGES, *estimators):
        times.setdefault(s, 0.0)
    last = int(keep[-1]) if len(keep) else -1
    start = time.perf_counter()
    for i in range(last + 1):
        t0 = time.perf_counter()
        if not cap.grab():
            raise RuntimeError(f"{path}: decoding stopped at frame {i} of {info.source_frames}")
        if i not in keep_set:
            times["decode"] += time.perf_counter() - t0
            continue
        q = keep_set[i]
        if pace_fps:
            times["decode"] += time.perf_counter() - t0
            wait = start + q / pace_fps - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
        ok, frame = cap.retrieve()
        read_time = time.time()
        times["decode"] += time.perf_counter() - t0
        if not ok:
            raise RuntimeError(f"{path}: cannot retrieve frame {i}")

        t0 = time.perf_counter()
        boxes = detect(frame)
        _sync()
        times["detect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        tracks = tracker.update(boxes, frame)
        times["track"] += time.perf_counter() - t0

        poses = {}
        for name, estimate in estimators.items():
            t0 = time.perf_counter()
            if len(tracks):
                poses[name] = estimate(
                    np.asarray(frame, np.uint8), tracks[:, :4].astype(np.float32)
                )
                _sync()
            times[name] += time.perf_counter() - t0
        yield Frame(q, read_time, len(boxes), np.asarray(tracks, np.float32), poses)
    cap.release()


def extract_video(
    path: Path,
    video_id: str,
    camera: str,
    cfg: dict[str, Any],  # Any: YAML config and JSON-ready stats
    estimators: Mapping[str, PoseEstimator] | None = None,
) -> tuple[dict[str, VideoPoses], dict[str, Any]]:
    """Extract pose records from one video with each estimator (default: the MMPose model
    of `cfg["pose"]`, named "pose"); return them and per-stage timings."""
    if estimators is None:
        pose = cfg["pose"]
        estimators = {"pose": mmpose_estimator(pose, cfg["device"], pose.get("flip_test"))}
    times: dict[str, float] = {}
    info = VideoInfo()
    rows = {name: Rows() for name in estimators}
    detections, tracked = [], []
    start = time.perf_counter()
    for fr in frame_stream(path, cfg, estimators, times, info):
        for name, (xy, scores) in fr.poses.items():
            rows[name].add(fr.q, fr.tracks, xy, scores)
        detections.append(fr.detections)
        tracked.append(len(fr.tracks))
    total = time.perf_counter() - start

    n_out = info.output_frames
    stats = {
        "source_fps": info.source_fps,
        "source_frames": info.source_frames,
        "source_size": list(info.source_size),
        "output_frames": n_out,
        "seconds": {**times, "total": total},
        "fps": {s: (n_out / v if v else None) for s, v in times.items()},
        "fps_total": n_out / total,
        "persons_per_frame": {
            "detections_mean": float(np.mean(detections)) if detections else 0.0,
            "tracks_mean": float(np.mean(tracked)) if tracked else 0.0,
            "tracks_max": int(np.max(tracked)) if tracked else 0,
        },
        "vram_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "vram_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
    }
    return {name: r.to_video(video_id, camera, n_out) for name, r in rows.items()}, stats
