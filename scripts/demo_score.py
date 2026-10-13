"""Session 2C: offline end-to-end demonstration on one theft video and one normal video.

Qualitative only: nothing here is a performance measurement. Reads the poses extracted in
.venv-pose, applies the C1 production treatment (taken from the C1 run's meta.json),
scores windows with S = 1, puts each window score on its last frame (operation mode,
ADR-0008) and takes the maximum over people. The provisional threshold is a quantile of
the C1 validation window scores (rule 2: no label is used).

Outputs: timeline PNG, summary JSON and skeleton-only video in the run; the video over
the real frames goes to `internal_dir` (data/, never versioned; ADR-0005).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from tcc.data.poselift import load_video
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.pipeline.resample import frames_to_keep
from tcc.pipeline.scoring import (
    alert_intervals,
    frame_max,
    last_frame_scores,
    load_extracted,
    overlaps,
    quantile_threshold,
    source_interval_to_output,
)
from tcc.preprocess.windows import video_windows
from tcc.utils.config import load_config
from tcc.utils.render import annotate_frame, blank_canvas
from tcc.utils.run_logger import start_run

SCORE_COLOR, INK, GRID = "#2a78d6", "#3d3d3a", "#e4e3df"
ALERT_COLOR, THEFT_COLOR = "#e34948", "#eb6834"


def _windows(video: Any, t: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return video_windows(video, t["T"], t["stride_eval"], t["max_gap"], t["conf_min"], t["center"])


def _timeline(
    path: Path,
    title: str,
    fs: np.ndarray,
    fps: float,
    threshold: float,
    alerts: list[tuple[int, int]],
    theft: list[tuple[int, int]],
) -> None:
    t = np.arange(len(fs)) / fps
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for a, b in alerts:
        ax.axvspan(a / fps, b / fps, color=ALERT_COLOR, alpha=0.15, linewidth=0)
    for a, b in theft:
        ax.axvspan(a / fps, b / fps, ymin=0, ymax=0.06, color=THEFT_COLOR, linewidth=0)
    ax.plot(t, fs, color=SCORE_COLOR, linewidth=1.5)
    ax.axhline(threshold, color=INK, linewidth=1, linestyle="--")
    ax.set_xlim(0, len(fs) / fps)
    ax.set_xlabel("seconds (15 fps)", color=INK)
    ax.set_ylabel("frame score (max over people)", color=INK)
    ax.set_title(title, color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], color=SCORE_COLOR, linewidth=1.5, label="frame score"),
        plt.Line2D([], [], color=INK, linestyle="--", label=f"threshold {threshold:.4f}"),
        Patch(color=ALERT_COLOR, alpha=0.15, label="alert (score > threshold)"),
    ]
    if theft:
        handles.append(Patch(color=THEFT_COLOR, label="annotated theft"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _render(
    video_cfg: dict[str, Any],
    poses: Any,
    keep: np.ndarray,
    row_scores: np.ndarray,
    fs: np.ndarray,
    threshold: float,
    theft: list[tuple[int, int]],
    fps: float,
    size: tuple[int, int],
    conf_min: float,
    out_public: Path,
    out_internal: Path,
) -> None:
    ps, is_ = video_cfg["public_scale"], video_cfg["internal_scale"]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    public = cv2.VideoWriter(str(out_public), fourcc, fps, (int(size[0] * ps), int(size[1] * ps)))
    internal = cv2.VideoWriter(
        str(out_internal), fourcc, fps, (int(size[0] * is_), int(size[1] * is_))
    )
    background = blank_canvas(size, ps)
    cap = cv2.VideoCapture(video_cfg["path"])
    keep_pos = {int(i): k for k, i in enumerate(keep)}
    for i in range(int(keep[-1]) + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot decode frame {i} of {video_cfg['path']}")
        if i not in keep_pos:
            continue
        q = keep_pos[i]
        rows = np.flatnonzero(poses.frame_idx == q)
        alert = bool(np.isfinite(fs[q]) and fs[q] > threshold)
        tag = " [furto anotado]" if any(a <= q < b for a, b in theft) else ""
        score = f"{fs[q]:.4f}" if np.isfinite(fs[q]) else "--"
        caption = f"{video_cfg['id']}  t={q / fps:.1f}s  score={score}  thr={threshold:.4f}{tag}"
        args = (
            poses.keypoints[rows],
            poses.track_id[rows],
            poses.bbox[rows],
            row_scores[rows],
            threshold,
        )
        public.write(annotate_frame(background, *args, ps, conf_min, alert, caption))
        real = cv2.resize(frame, (int(size[0] * is_), int(size[1] * is_)))
        internal.write(annotate_frame(real, *args, is_, conf_min, alert, caption))
    public.release()
    internal.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/demo_score.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    c1_run, extraction_run = Path(config["c1_run"]), Path(config["extraction_run"])
    c1_config = json.loads((c1_run / "meta.json").read_text(encoding="utf-8"))["config"]
    treatment, dataset = c1_config["treatment"], c1_config["dataset"]
    extraction = json.loads((extraction_run / "results.json").read_text(encoding="utf-8"))
    fps = float(
        json.loads((extraction_run / "meta.json").read_text(encoding="utf-8"))["config"][
            "target_fps"
        ]
    )
    config["c1_treatment"], config["c1_dataset"] = treatment, dataset
    models = {"autoencoder": {"name": "C1 ST-GCN autoencoder", "version": c1_run.name,
                              "weights": str(c1_run / "model.pt")}}  # fmt: skip
    run_dir = start_run("demo_2c", config=config, seeds={}, models=models)

    ck = torch.load(c1_run / "model.pt", map_location="cpu", weights_only=True)
    m = ck["model"]
    model = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
    model.load_state_dict(ck["state"], strict=True)
    device = torch.device(config["device"])
    model.to(device).eval()
    loss_norm, batch = c1_config["loss"]["loss_norm"], config["batch_size"]

    root = Path(dataset["root"])
    val = [
        window_scores(model, _windows(load_video(root, "train", v, dataset), treatment)[0],
                      device, loss_norm, batch)
        for v in ck["val_ids"]
    ]  # fmt: skip
    val_scores = np.concatenate(val)
    q = config["threshold_quantile"]
    threshold = quantile_threshold(val_scores, q)
    origin = {
        "quantile": q,
        "of": "C1 validation window scores, S = 1, production treatment, normal only",
        "c1_run": str(c1_run),
        "val_videos": len(ck["val_ids"]),
        "val_windows": int(len(val_scores)),
        "value": threshold,
    }
    print(f"threshold {threshold:.6f} (q={q} of {len(val_scores)} validation windows)")

    Path(config["internal_dir"]).mkdir(parents=True, exist_ok=True)
    for video_cfg in config["videos"]:
        vid = video_cfg["id"]
        if not isinstance(vid, str):
            raise SystemExit(f"video ids must be quoted strings, got {vid!r}")
        ext = extraction[vid]
        poses = load_extracted(extraction_run / f"{vid}_poses.npz", vid, video_cfg["camera"])

        start = time.perf_counter()
        x, meta = _windows(poses, treatment)
        t_treatment = time.perf_counter() - start
        start = time.perf_counter()
        scores = window_scores(model, x, device, loss_norm, batch)
        t_model = time.perf_counter() - start

        per_person = last_frame_scores(meta, scores, treatment["T"])
        fs = frame_max(per_person, poses.n_frames)
        pairs = zip(poses.track_id.tolist(), poses.frame_idx.tolist(), strict=True)
        row_scores = np.array([per_person.get(pair, np.nan) for pair in pairs])
        alerts = alert_intervals(fs, threshold)
        keep = frames_to_keep(ext["source_fps"], ext["source_frames"], fps)[: poses.n_frames]
        theft = [
            iv
            for a, b in video_cfg["theft_source_frames"]
            if (iv := source_interval_to_output(keep, a, b)) is not None
        ]

        scored = np.isfinite(fs)
        above = scored & (fs > threshold)
        n = poses.n_frames
        summary: dict[str, Any] = {
            "demonstration_only": True,
            "video": vid,
            "source": {k: ext[k] for k in ["source_fps", "source_frames", "source_size"]},
            "output_frames": n,
            "seconds": n / fps,
            "threshold": origin,
            "windows": int(len(x)),
            "frames_with_score": int(scored.sum()),
            "alerts": len(alerts),
            "alert_intervals_seconds": [[a / fps, b / fps] for a, b in alerts],
            "fraction_of_time_above_threshold": float(above.sum() / n),
            "fraction_of_scored_frames_above_threshold": (
                float(above.sum() / scored.sum()) if scored.any() else None
            ),
            "fps": {
                **{f"extract_{k}": v for k, v in ext["fps"].items()},
                "extract_total": ext["fps_total"],
                "treatment": n / t_treatment,
                "model": n / t_model,
            },
            "persons_per_frame": ext["persons_per_frame"],
        }
        if video_cfg["theft_source_frames"]:
            theft_frames = np.zeros(n, bool)
            for a, b in theft:
                theft_frames[a:b] = True
            summary["theft"] = {
                "annotation_source_frames": video_cfg["theft_source_frames"],
                "annotation_source": video_cfg["annotation_source"],
                "output_intervals": theft,
                "output_intervals_seconds": [[a / fps, b / fps] for a, b in theft],
                "alert_overlaps_annotation": any(overlaps(al, th) for al in alerts for th in theft),
                "alerts_outside_annotation": sum(
                    1 for al in alerts if not any(overlaps(al, th) for th in theft)
                ),
                "fraction_of_theft_frames_above_threshold": float(
                    above[theft_frames].sum() / max(theft_frames.sum(), 1)
                ),
                "fraction_of_non_theft_frames_above_threshold": float(
                    above[~theft_frames].sum() / max((~theft_frames).sum(), 1)
                ),
            }
        (run_dir / f"{vid}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        _timeline(run_dir / f"{vid}_timeline.png", f"{vid} (demonstration, not a metric)",
                  fs, fps, threshold, alerts, theft)  # fmt: skip
        _render(video_cfg, poses, keep, row_scores, fs, threshold, theft, fps,
                tuple(ext["source_size"]), config["conf_min_draw"],
                run_dir / f"{vid}_public.mp4",
                Path(config["internal_dir"]) / f"{vid}_internal.mp4")  # fmt: skip
        brief = {k: summary[k] for k in ["alerts", "fraction_of_time_above_threshold"]}
        print(vid, json.dumps(brief), json.dumps(summary.get("theft", {})))
    print(run_dir)


if __name__ == "__main__":
    main()
