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

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from tcc.data.poselift import load_video_with_real
from tcc.eval.alerts import alerts, runs_above
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.pipeline.resample import frames_to_keep
from tcc.pipeline.scoring import (
    NOT_SCORABLE,
    SCORED,
    STATES,
    frame_scores,
    last_frame_reasons,
    last_frame_scores,
    load_extracted,
    overlaps,
    quantile_threshold,
    source_interval_to_output,
)
from tcc.preprocess.quality import REASONS, SCORABLE
from tcc.preprocess.windows import treated_windows
from tcc.utils.config import load_config
from tcc.utils.demo_video import people_by_frame, render_videos, timeline
from tcc.utils.run_logger import start_run


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
        timeline(run_dir / f"{vid}_timeline.png", f"{vid} (demonstration, not a metric)",
                  fs, fps, threshold, final, theft)  # fmt: skip
        people = people_by_frame(poses, treatment)
        render_videos(video_cfg, people, keep, per_person, per_reason, fs,
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
