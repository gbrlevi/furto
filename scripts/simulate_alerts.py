"""Session 3, task C6: alerts per hour on the C1 validation videos versus the threshold.

Validation only (rules 1 and 2): the videos are the validation part of the PoseLift
training partition stored in each checkpoint. Operation mode: causal last-frame score,
maximum over people, no smoothing, alerts of at least `min_alert_frames` merged when less
than `merge_gap_frames` apart. Frames with nobody or only not-scorable windows never
alert. With about a quarter of an hour of video the curve is a coarse estimate: one alert
is already several alerts per hour.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tcc.data.poselift import load_video_with_real
from tcc.eval.alerts import alerts
from tcc.eval.threshold import alert_rate_curve, candidate_thresholds, threshold_for_rate
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.pipeline.scoring import (
    NOT_SCORABLE,
    SCORED,
    frame_scores,
    last_frame_reasons,
    last_frame_scores,
)
from tcc.preprocess.quality import SCORABLE
from tcc.preprocess.windows import treated_windows
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run

LINE, INK, GRID, MUTED = "#2a78d6", "#3d3d3a", "#e4e3df", "#8a8984"


def _validation(run: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Operational frame scores of the validation videos of one trained model."""
    c1 = json.loads((run / "meta.json").read_text(encoding="utf-8"))["config"]
    t, dataset = c1["treatment"], c1["dataset"]
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    m = ck["model"]
    model = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
    model.load_state_dict(ck["state"], strict=True)
    device = torch.device(config["device"])
    model.to(device).eval()
    root = Path(dataset["root"])
    size = (int(dataset["image_size"][0]), int(dataset["image_size"][1]))
    videos, rows = [], []
    for vid in ck["val_ids"]:
        video, real = load_video_with_real(root, "train", vid, dataset)
        w = treated_windows(video, t, t["stride_eval"], size, real)
        ok = w.reason == SCORABLE
        scores = window_scores(
            model, w.x[ok], device, c1["loss"]["loss_norm"], config["batch_size"]
        )
        fs = frame_scores(
            last_frame_scores(w.meta[ok], scores, t["T"]),
            last_frame_reasons(w.meta, w.reason, t["T"]),
            video.n_frames,
        )
        videos.append((fs.score, fs.state == SCORED))
        rows.append(
            {
                "video": vid,
                "frames": video.n_frames,
                "windows": int(len(w.meta)),
                "windows_not_scorable": int((~ok).sum()),
                "frames_scored": int((fs.state == SCORED).sum()),
                "frames_not_scorable": int((fs.state == NOT_SCORABLE).sum()),
            }
        )
    return {"videos": videos, "rows": rows, "quality_gate": t["quality_gate"], "run": str(run)}


def _plot(curves: dict[str, dict[str, Any]], targets: list[float], path: Path) -> None:
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 3.8))
    for ax, (name, c) in zip(np.atleast_1d(axes), curves.items(), strict=True):
        ax.step(c["thresholds"], c["rates"], where="post", color=LINE, linewidth=2)
        for n in targets:
            ax.axhline(n, color=MUTED, linewidth=0.8, linestyle="--")
            ax.text(c["thresholds"][-1], n, f" N = {n:g}", color=INK, fontsize=8, va="center")
        step = 1 / c["hours"]
        ax.axhline(step, color=INK, linewidth=0.8, linestyle=":")
        ax.text(c["thresholds"][0], step, f"1 alert = {step:.1f}/h ", color=INK, fontsize=8,
                va="bottom")  # fmt: skip
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(bottom=0)
        ax.set_xscale("log")
        ax.set_xlabel("threshold (reconstruction error, log)", color=INK)
        ax.set_ylabel("alerts per hour (symlog)", color=INK)
        ax.set_title(f"{name}: {c['hours'] * 60:.1f} min of validation", color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle("Validation alerts per hour vs threshold (only normal behaviour)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulate_alerts.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    models = {
        name: {"name": "C1 ST-GCN autoencoder", "version": Path(run).name,
               "weights": str(Path(run) / "model.pt")}
        for name, run in config["models"].items()
    }  # fmt: skip
    run_dir = start_run("simulate_alerts", config=config, seeds={}, models=models)
    fps = 15
    min_frames, merge = config["min_alert_frames"], config["merge_gap_frames"]
    cand = config["candidates"]

    results: dict[str, Any] = {}
    curves = {}
    for name, run in config["models"].items():
        val = _validation(Path(run), config)
        videos = val["videos"]
        sample = np.concatenate([s[m] for s, m in videos])
        candidates = candidate_thresholds(sample, cand["low"], cand["high"], cand["n"])
        rates = alert_rate_curve(videos, candidates, min_frames, merge, fps)
        frames = sum(len(s) for s, _ in videos)
        hours = frames / fps / 3600
        targets = {}
        for n in config["target_rates"]:
            thr, reached = threshold_for_rate(videos, n, candidates, min_frames, merge, fps)
            found = [alerts(s, m, thr, min_frames, merge) for s, m in videos]
            targets[str(n)] = {
                "threshold": thr,
                "reached": reached,
                "alerts": sum(len(a) for a in found),
                "alerts_per_hour": sum(len(a) for a in found) / hours,
                "quantile_of_scored_frames": float((sample <= thr).mean()),
            }
        with open(run_dir / f"curve_{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["threshold", "alerts_per_hour", "alerts"])
            for thr, r in zip(candidates.tolist(), rates.tolist(), strict=True):
                writer.writerow([thr, r, round(r * hours)])
        curves[name] = {"thresholds": candidates, "rates": rates, "hours": hours}
        results[name] = {
            "run": val["run"],
            "quality_gate": val["quality_gate"],
            "videos": len(videos),
            "frames": frames,
            "hours": hours,
            "one_alert_per_hour": 1 / hours,
            "frames_scored": int(sum(m.sum() for _, m in videos)),
            "max_rate": float(rates.max()),
            "candidates": int(len(candidates)),
            "targets": targets,
            "per_video": val["rows"],
        }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot(curves, [float(n) for n in config["target_rates"]], run_dir / "alert_rate_curve.png")
    print(json.dumps({k: {kk: v[kk] for kk in ["hours", "frames_scored", "targets"]}
                      for k, v in results.items()}, indent=1))  # fmt: skip
    print(run_dir)


if __name__ == "__main__":
    main()
