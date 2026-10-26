"""Offline end-to-end demonstration on public videos (sessions 2C and 2D).

Qualitative only: nothing here is a performance measurement. Reads the poses extracted in
.venv-pose, applies the treatment of the C1 run (from its meta.json, including the scale
guard of ADR-0015 and the quality gate of ADR-0014 when they are on), scores windows with
S = 1, puts each window score on its last frame (operation mode, ADR-0008) and takes the
maximum over people. Frames with only filtered windows are "not scorable". The
provisional threshold is a quantile of the C1 validation scores of scorable windows
(rule 2: no label is used). Alerts follow `tcc.eval.alerts` (minimum duration, merging).

Outputs: timeline PNG, summary JSON, per-frame event log CSV and skeleton-only video in
the run; the video over the real frames goes to `internal_dir` (data/, never versioned;
ADR-0005).
"""

import argparse
import csv
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

from tcc.data.poselift import load_video_with_real
from tcc.eval.alerts import Alert, alerts, runs_above
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.pipeline.resample import frames_to_keep
from tcc.pipeline.scoring import (
    NOT_SCORABLE,
    SCORED,
    STATES,
    FrameScores,
    frame_scores,
    last_frame_reasons,
    last_frame_scores,
    load_extracted,
    overlaps,
    quantile_threshold,
    source_interval_to_output,
)
from tcc.preprocess.quality import CRITERIA, REASONS, SCORABLE
from tcc.preprocess.trajectories import split_trajectories, tracks
from tcc.preprocess.windows import treated_windows
from tcc.utils.config import load_config
from tcc.utils.render import annotate_frame, blank_canvas
from tcc.utils.run_logger import start_run

SCORE_COLOR, INK, GRID = "#2a78d6", "#3d3d3a", "#e4e3df"
ALERT_COLOR, THEFT_COLOR = "#e34948", "#eb6834"
# Not-scorable shading, one categorical slot per criterion in CRITERIA order
REASON_COLORS = dict(
    zip(CRITERIA, ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"], strict=True)
)


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _timeline(
    path: Path, title: str, fs: FrameScores, fps: float, threshold: float,
    final: list[Alert], theft: list[tuple[int, int]],
) -> None:  # fmt: skip
    n = len(fs.score)
    t = np.arange(n) / fps
    fig, ax = plt.subplots(figsize=(10, 3.6))
    handles = [
        plt.Line2D([], [], color=SCORE_COLOR, linewidth=1.5, label="frame score"),
        plt.Line2D([], [], color=INK, linestyle="--", label=f"threshold {threshold:.4f}"),
        Patch(color=ALERT_COLOR, alpha=0.18, label="alert (>= 15 frames, merged < 2 s)"),
    ]
    for r, name in REASONS.items():
        mask = (fs.state == NOT_SCORABLE) & (fs.reason == r)
        if r == SCORABLE or not mask.any():
            continue
        for a, b in _spans(mask):
            ax.axvspan(a / fps, b / fps, color=REASON_COLORS[name], alpha=0.25, linewidth=0)
        handles.append(Patch(color=REASON_COLORS[name], alpha=0.25, label=f"not scorable: {name}"))
    for al in final:
        ax.axvspan(al.start / fps, al.end / fps, color=ALERT_COLOR, alpha=0.18, linewidth=0)
    for a, b in theft:
        ax.axvspan(a / fps, b / fps, ymin=0, ymax=0.05, color=THEFT_COLOR, linewidth=0)
    if theft:
        handles.append(Patch(color=THEFT_COLOR, label="annotated theft"))
    ax.plot(t, np.where(fs.state == SCORED, fs.score, np.nan), color=SCORE_COLOR, linewidth=1.5)
    ax.axhline(threshold, color=INK, linewidth=1, linestyle="--")
    ax.set_yscale("symlog", linthresh=threshold)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, n / fps)
    ax.set_xlabel("seconds (15 fps)", color=INK)
    ax.set_ylabel("frame score (symlog)", color=INK)
    ax.set_title(title, color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.0, 1.0))  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _people(poses: Any, t: dict[str, Any]) -> dict[int, list[tuple[Any, ...]]]:
    """Per output frame: (track, keypoints, box, interpolated) of every detection, plus the
    poses the treatment filled inside trajectory gaps (drawn as interpolated)."""
    out: dict[int, list[tuple[Any, ...]]] = {}
    for i in range(len(poses)):
        q = int(poses.frame_idx[i])
        out.setdefault(q, []).append(
            (int(poses.track_id[i]), poses.keypoints[i], poses.bbox[i], False)
        )
    for tr in split_trajectories(tracks(poses), t["max_gap"], t["T"]):
        assert tr.real is not None and tr.boxes is not None
        for k in np.flatnonzero(~tr.real):
            q = int(tr.frames[k])
            out.setdefault(q, []).append((tr.track_id, tr.keypoints[k], tr.boxes[k], True))
    return out


def _render(
    video_cfg: dict[str, Any], people: dict[int, list[tuple[Any, ...]]], keep: np.ndarray,
    per_person: dict[tuple[int, int], float], per_reason: dict[tuple[int, int], int],
    fs: FrameScores, alert_frames: np.ndarray, threshold: float, theft: list[tuple[int, int]],
    fps: float, size: tuple[int, int], conf_min: float, out_public: Path, out_internal: Path,
) -> None:  # fmt: skip
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
        rows = people.get(q, [])
        tids = np.array([r[0] for r in rows], np.int64)
        kps = np.array([r[1] for r in rows], np.float32).reshape(-1, 17, 3)
        boxes = np.array([r[2] for r in rows], np.float32).reshape(-1, 4)
        interp = np.array([r[3] for r in rows], np.bool_)
        scores = np.array([per_person.get((int(k), q), np.nan) for k in tids])
        reasons = [REASONS[per_reason[(int(k), q)]] if (int(k), q) in per_reason else ""
                   for k in tids]  # fmt: skip
        tag = " [furto anotado]" if any(a <= q < b for a, b in theft) else ""
        state = int(fs.state[q])
        score = f"{fs.score[q]:.4f}" if state == SCORED else STATES[state]
        caption = f"{video_cfg['id']}  t={q / fps:.1f}s  score={score}  thr={threshold:.4f}{tag}"
        args = (kps, tids, boxes, scores, threshold)
        extra = {"interpolated": interp, "reasons": reasons}
        alert = bool(alert_frames[q])
        public.write(annotate_frame(background, *args, ps, conf_min, alert, caption, **extra))
        real = cv2.resize(frame, (int(size[0] * is_), int(size[1] * is_)))
        internal.write(annotate_frame(real, *args, is_, conf_min, alert, caption, **extra))
    public.release()
    internal.release()


def _theft_stats(alert_list: list[tuple[int, int]], theft: list[tuple[int, int]]) -> dict[str, int]:
    inside = sum(1 for al in alert_list if any(overlaps(al, th) for th in theft))
    return {"alerts": len(alert_list), "overlapping_annotation": inside,
            "outside_annotation": len(alert_list) - inside}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/demo_score_2d.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    c1_run = Path(config["c1_run"])
    c1_config = json.loads((c1_run / "meta.json").read_text(encoding="utf-8"))["config"]
    treatment, dataset = c1_config["treatment"], c1_config["dataset"]
    config["c1_treatment"], config["c1_dataset"] = treatment, dataset
    models = {"autoencoder": {"name": "C1 ST-GCN autoencoder", "version": c1_run.name,
                              "weights": str(c1_run / "model.pt")}}  # fmt: skip
    run_dir = start_run(config["run_name"], config=config, seeds={}, models=models)

    ck = torch.load(c1_run / "model.pt", map_location="cpu", weights_only=True)
    m = ck["model"]
    model = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
    model.load_state_dict(ck["state"], strict=True)
    device = torch.device(config["device"])
    model.to(device).eval()
    loss_norm, batch = c1_config["loss"]["loss_norm"], config["batch_size"]
    T, stride = treatment["T"], treatment["stride_eval"]

    root = Path(dataset["root"])
    pl_size = (int(dataset["image_size"][0]), int(dataset["image_size"][1]))
    val, val_total = [], 0
    for v in ck["val_ids"]:
        video, real = load_video_with_real(root, "train", v, dataset)
        w = treated_windows(video, treatment, stride, pl_size, real)
        val.append(window_scores(model, w.x[w.reason == SCORABLE], device, loss_norm, batch))
        val_total += len(w.x)
    val_scores = np.concatenate(val)
    q = config["threshold_quantile"]
    threshold = quantile_threshold(val_scores, q)
    origin = {
        "quantile": q,
        "of": "C1 validation scores of scorable windows, S = 1, production treatment, normal only",
        "c1_run": str(c1_run),
        "val_videos": len(ck["val_ids"]),
        "val_windows": val_total,
        "val_scorable_windows": int(len(val_scores)),
        "value": threshold,
    }
    print(f"threshold {threshold:.6f} (q={q} of {len(val_scores)} scorable validation windows)")

    min_frames, merge_gap = config["min_alert_frames"], config["merge_gap_frames"]
    before_after = {}
    Path(config["internal_dir"]).mkdir(parents=True, exist_ok=True)
    for video_cfg in config["videos"]:
        vid = video_cfg["id"]
        if not isinstance(vid, str):
            raise SystemExit(f"video ids must be quoted strings, got {vid!r}")
        extraction_run = Path(video_cfg["extraction_run"])
        ext = json.loads((extraction_run / "results.json").read_text(encoding="utf-8"))[vid]
        fps = float(
            json.loads((extraction_run / "meta.json").read_text(encoding="utf-8"))["config"][
                "target_fps"
            ]
        )
        poses = load_extracted(extraction_run / f"{vid}_poses.npz", vid, video_cfg["camera"])
        size = (int(ext["source_size"][0]), int(ext["source_size"][1]))

        start = time.perf_counter()
        w = treated_windows(poses, treatment, stride, size)
        t_treatment = time.perf_counter() - start
        ok = w.reason == SCORABLE
        start = time.perf_counter()
        scores = window_scores(model, w.x[ok], device, loss_norm, batch)
        t_model = time.perf_counter() - start

        n = poses.n_frames
        per_person = last_frame_scores(w.meta[ok], scores, T)
        per_reason = last_frame_reasons(w.meta, w.reason, T)
        fs = frame_scores(per_person, per_reason, n)
        scored = fs.state == SCORED
        raw = runs_above(fs.score, scored, threshold)
        final = alerts(fs.score, scored, threshold, min_frames, merge_gap)
        alert_frames = np.zeros(n, np.bool_)
        for al in final:
            alert_frames[al.start : al.end] = True
        keep = frames_to_keep(ext["source_fps"], ext["source_frames"], fps)[:n]
        theft = [
            iv
            for a, b in video_cfg["theft_source_frames"]
            if (iv := source_interval_to_output(keep, a, b)) is not None
        ]

        above = scored & (fs.score > threshold)
        not_scorable = fs.state == NOT_SCORABLE
        frame_reasons = fs.reason[not_scorable]
        reason_counts = {REASONS[r]: int((frame_reasons == r).sum()) for r in range(1, 6)}
        window_reasons = {REASONS[r]: int((w.reason == r).sum()) for r in REASONS}
        summary: dict[str, Any] = {
            "demonstration_only": True,
            "video": vid,
            "source": {k: ext[k] for k in ["source_fps", "source_frames", "source_size"]},
            "output_frames": n,
            "seconds": n / fps,
            "threshold": origin,
            "windows": int(len(w.x)),
            "windows_by_reason": window_reasons,
            "frames_with_score": int(scored.sum()),
            "frames_not_scorable": int(not_scorable.sum()),
            "fraction_of_frames_not_scorable": float(not_scorable.mean()),
            "fraction_of_frames_with_a_window_not_scorable": (
                float(not_scorable.sum() / (scored | not_scorable).sum())
                if (scored | not_scorable).any()
                else None
            ),
            "not_scorable_frames_by_reason": reason_counts,
            "most_frequent_reason": (
                max(reason_counts, key=lambda k: reason_counts[k]) if not_scorable.any() else None
            ),
            "alerts_without_min_duration": {
                "count": len(raw),
                "intervals_seconds": [[a / fps, b / fps] for a, b in raw],
            },
            "alerts_with_min_duration": {
                "min_alert_frames": min_frames,
                "merge_gap_frames": merge_gap,
                "count": len(final),
                "intervals_seconds": [[al.start / fps, al.end / fps] for al in final],
                "fired_seconds": [al.fired / fps for al in final],
            },
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
                "without_min_duration": _theft_stats(raw, theft),
                "with_min_duration": _theft_stats([(al.start, al.end) for al in final], theft),
                "fraction_of_theft_frames_above_threshold": float(
                    above[theft_frames].sum() / max(theft_frames.sum(), 1)
                ),
                "fraction_of_non_theft_frames_above_threshold": float(
                    above[~theft_frames].sum() / max((~theft_frames).sum(), 1)
                ),
                "fraction_of_theft_frames_not_scorable": float(
                    not_scorable[theft_frames].sum() / max(theft_frames.sum(), 1)
                ),
            }
        (run_dir / f"{vid}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        with open(run_dir / f"{vid}_events.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "seconds", "score", "state", "reason", "alert"])
            for k in range(n):
                writer.writerow([
                    k, f"{k / fps:.3f}", f"{fs.score[k]:.6g}" if scored[k] else "",
                    STATES[int(fs.state[k])], REASONS[int(fs.reason[k])] if fs.reason[k] else "",
                    int(alert_frames[k]),
                ])  # fmt: skip
        _timeline(run_dir / f"{vid}_timeline.png", f"{vid} (demonstration, not a metric)",
                  fs, fps, threshold, final, theft)  # fmt: skip
        _render(video_cfg, _people(poses, treatment), keep, per_person, per_reason, fs,
                alert_frames, threshold, theft, fps, size, config["conf_min_draw"],
                run_dir / f"{vid}_public.mp4",
                Path(config["internal_dir"]) / f"{vid}_internal.mp4")  # fmt: skip

        before = json.loads(Path(video_cfg["before_summary"]).read_text(encoding="utf-8"))
        bt = before.get("theft", {})
        st = summary.get("theft", {})
        before_after[vid] = {
            "before": {
                "summary": video_cfg["before_summary"],
                "threshold": before["threshold"]["value"],
                "alerts": before["alerts"],
                "alerts_overlapping_theft": (
                    before["alerts"] - bt["alerts_outside_annotation"] if bt else None
                ),
                "theft_frames_above": bt.get("fraction_of_theft_frames_above_threshold"),
                "non_theft_frames_above": bt.get("fraction_of_non_theft_frames_above_threshold"),
                "fraction_of_time_above": before["fraction_of_time_above_threshold"],
            },
            "after": {
                "threshold": threshold,
                "alerts_without_min_duration": len(raw),
                "alerts_with_min_duration": len(final),
                "alerts_overlapping_theft_without_min": (
                    st["without_min_duration"]["overlapping_annotation"] if st else None
                ),
                "alerts_overlapping_theft_with_min": (
                    st["with_min_duration"]["overlapping_annotation"] if st else None
                ),
                "theft_frames_above": st.get("fraction_of_theft_frames_above_threshold"),
                "non_theft_frames_above": st.get("fraction_of_non_theft_frames_above_threshold"),
                "fraction_of_time_above": summary["fraction_of_time_above_threshold"],
                "fraction_of_frames_not_scorable": summary["fraction_of_frames_not_scorable"],
                "most_frequent_reason": summary["most_frequent_reason"],
            },
        }
        print(vid, json.dumps(before_after[vid]["after"]))
    (run_dir / "before_after.json").write_text(json.dumps(before_after, indent=2), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
