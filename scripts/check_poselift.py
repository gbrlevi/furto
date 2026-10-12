"""Inspect the PoseLift download: counts, split, NaN confidences and training statistics.

Counts and statistics are computed in tcc.data.stats; this script loads, records the run
and writes results.json plus figures. Keypoints are read exactly as stored (no axis swap,
no NaN replacement), because the NaN and axis measurements need the raw values.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tcc.data.poselift import load_video
from tcc.data.schema import VideoPoses
from tcc.data.skeleton import JOINT_NAMES
from tcc.data.split import official_split
from tcc.data.stats import (
    axis_evidence,
    confidence_histograms,
    frame_counts,
    nan_rows_vs_fractional_boxes,
    nan_runs,
    nan_summary,
    track_lengths,
)
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run

BLUE = "#2a78d6"
INK = "#3d3d3a"
GRID = "#e4e3df"


def _style(ax: Any) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7)


def _plot_confidence(hist: np.ndarray, bins: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(13, 6), sharex=True)
    width = bins[1] - bins[0]
    for j, ax in enumerate(axes.flat):
        if j >= len(JOINT_NAMES):
            ax.axis("off")
            continue
        ax.bar(
            bins[:-1], hist[j] / max(hist[j].sum(), 1), width=width * 0.9, align="edge", color=BLUE
        )
        ax.set_title(JOINT_NAMES[j], fontsize=8, color=INK)
        _style(ax)
    fig.suptitle("PoseLift train: finite confidence per joint (fraction of points)", color=INK)
    fig.supxlabel("confidence", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_lengths(observed: np.ndarray, thresholds: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    edges = np.logspace(0, np.log10(observed.max() + 1), 40)
    ax.hist(observed, bins=edges, color=BLUE, rwidth=0.9)
    ax.set_xscale("log")
    for t in thresholds:
        ax.axvline(t, color=INK, linewidth=0.8, linestyle="--")
        ax.text(t, ax.get_ylim()[1] * 0.95, f" T={t}", fontsize=7, color=INK, va="top")
    ax.set_xlabel("frames observed per track (log scale)", color=INK)
    ax.set_ylabel("tracks", color=INK)
    ax.set_title("PoseLift train: track length", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _partition(videos: list[VideoPoses], thresholds: list[int]) -> dict[str, Any]:
    lengths = track_lengths(videos)
    observed = lengths["observed"]
    return {
        "videos": len(videos),
        "cameras": dict(sorted(Counter(v.camera for v in videos).items())),
        "tracks": int(len(observed)),
        "person_frames": int(sum(len(v) for v in videos)),
        "frame_counts": frame_counts(videos),
        "videos_with_theft": sum(
            1 for v in videos if v.frame_labels is not None and v.frame_labels.any()
        ),
        "track_observed_frames": {
            "min": int(observed.min()),
            "median": float(np.median(observed)),
            "max": int(observed.max()),
            "fraction_below": {t: float((observed < t).mean()) for t in thresholds},
        },
        "gap_free_runs_fraction_below": {
            t: float((lengths["runs"] < t).mean()) for t in thresholds
        },
        "gaps_inside_tracks": {
            "count": int(len(lengths["gaps"])),
            "missing_frames": int(lengths["gaps"].sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/check_poselift.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = load_config(Path(config["dataset_config"]))
    config["dataset"] = dataset
    run_dir = start_run("check_poselift", config=config, seeds={})

    root = Path(dataset["root"])
    raw_cfg = {"keypoint_order": "xy", "bbox_format": "xyxy", "conf_nan": None}
    split = official_split(root)
    videos = {
        part: [load_video(root, part, i, raw_cfg) for i in ids] for part, ids in split.items()
    }
    thresholds = config["short_track_thresholds"]
    everything = videos["train"] + videos["test"]

    all_kp = np.concatenate([v.keypoints for v in everything])
    all_box = np.concatenate([v.bbox for v in everything])
    runs = [r for v in everything for r in nan_runs(v, config["interpolation_tolerance_px"])]
    interior = [r for r in runs if r["interior"]]
    deviations = np.array([r["max_deviation_px"] for r in interior])
    bins = np.linspace(0.0, 1.0, config["confidence_bins"] + 1)
    train_kp = np.concatenate([v.keypoints for v in videos["train"]])
    hist = confidence_histograms(train_kp, bins)

    results: dict[str, Any] = {
        "split_source": "Pickle_files/Train and Pickle_files/Test folders",
        "partitions": {p: _partition(vs, thresholds) for p, vs in videos.items()},
        "published": config["published"],
        "cameras_total": sorted({v.camera for v in everything}),
        "label_frames_after_last_pose": {
            v.video_id: {"labels": v.n_frames, "last_pose_frame": int(v.frame_idx.max())}
            for v in videos["test"]
            if v.n_frames != int(v.frame_idx.max()) + 1
        },
        "nan": {
            "all": nan_summary(all_kp),
            "train": nan_summary(train_kp),
            "vs_fractional_boxes": nan_rows_vs_fractional_boxes(everything),
            "all_nan_rows_runs": {
                "runs": len(runs),
                "rows": int(sum(r["length"] for r in runs)),
                "interior_runs": len(interior),
                "interior_rows": int(sum(r["length"] for r in interior)),
                "edge_runs": len(runs) - len(interior),
                "interior_linear_runs": int(sum(r["linear"] for r in interior)),
                "interior_max_deviation_px_percentiles": {
                    p: float(np.percentile(deviations, p)) for p in (50, 90, 99, 100)
                }
                if len(deviations)
                else {},
                "run_length_counts": dict(sorted(Counter(int(r["length"]) for r in runs).items())),
            },
        },
        "axis_evidence": axis_evidence(all_kp, all_box),
        "confidence_histogram_train": {"bins": bins.tolist(), "counts": hist.tolist()},
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_confidence(hist, bins, run_dir / "confidence_per_joint_train.png")
    _plot_lengths(
        track_lengths(videos["train"])["observed"], thresholds, run_dir / "track_length_train.png"
    )

    published = config["published"]
    print(run_dir)
    print(f"{'convention':<22}{'train normal':>14}{'test normal':>13}{'test theft':>12}")
    for c in ["all_frames", "frames_with_person", "person_frames"]:
        tr = results["partitions"]["train"]["frame_counts"][c]
        te = results["partitions"]["test"]["frame_counts"][c]
        print(f"{c:<22}{tr['normal']:>14,}{te['normal']:>13,}{te['theft']:>12,}")
    pub = f"{published['train']['normal']:>14,}{published['test']['normal']:>13,}"
    print(f"{'published (Table 1)':<22}{pub}{published['test']['theft']:>12,}")


if __name__ == "__main__":
    main()
