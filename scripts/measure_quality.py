"""Session 2D, task 3: quality-gate thresholds (ADR-0014) and scale-guard ratios (ADR-0015).

Training part of the PoseLift train partition only (same split as C1). Windows with
T = 24, S = 6 and the current production treatment. Each threshold follows the rule
declared in the config before measuring; nothing is tuned afterwards. The removal
fraction is then measured on the training windows and on the validation windows.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tcc.data.poselift import load_video_with_real
from tcc.data.split import official_split, split_train_val
from tcc.preprocess.quality import (
    CRITERIA,
    REASONS,
    failures,
    gate_reasons,
    torso_box_ratio,
    window_quality,
)
from tcc.preprocess.trajectories import split_trajectories, tracks
from tcc.preprocess.windows import raw_windows
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run

BAR, INK, GRID = "#2a78d6", "#3d3d3a", "#e4e3df"


def _measure(
    root: Path, ids: list[str], dataset: dict[str, Any], t: dict[str, Any], stride: int,
    size: tuple[int, int], margin: float,
) -> dict[str, Any]:  # fmt: skip
    metrics: dict[str, list[np.ndarray]] = {k: [] for k in CRITERIA}
    ratios, outside, box_rows = [], 0, 0
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for vid in ids:
        video, real = load_video_with_real(root, "train", vid, dataset)
        b = video.bbox
        outside += int(
            ((b[:, 0] < 0) | (b[:, 1] < 0) | (b[:, 2] > size[0]) | (b[:, 3] > size[1])).sum()
        )
        box_rows += len(b)
        if len(b):
            lo, hi = np.minimum(lo, b[:, :2].min(0)), np.maximum(hi, b[:, 2:].max(0))
        trajs = split_trajectories(tracks(video, real), t["max_gap"], t["T"])
        w = raw_windows(trajs, t["T"], stride)
        for k, v in window_quality(
            w.keypoints, w.boxes, w.real, size, t["conf_min"], margin
        ).items():
            metrics[k].append(v)
        for tr in trajs:  # distinct frames covered by the windows
            assert tr.boxes is not None
            ratios.append(torso_box_ratio(tr.keypoints, tr.boxes, t["conf_min"]))
    return {
        "metrics": {k: np.concatenate(v) for k, v in metrics.items()},
        "ratios": np.concatenate(ratios),
        "boxes": {
            "rows": box_rows,
            "rows_outside_image": outside,
            "min_x1_y1": lo.tolist(),
            "max_x2_y2": hi.tolist(),
        },
    }


def _removal(metrics: dict[str, np.ndarray], thresholds: dict[str, float]) -> dict[str, Any]:
    reason = gate_reasons(metrics, thresholds)
    n = len(reason)
    return {
        "windows": n,
        "removed": int((reason != 0).sum()),
        "removed_fraction": float((reason != 0).mean()),
        "failing_each_criterion": {
            k: float(f.mean()) for k, f in failures(metrics, thresholds).items()
        },
        "by_first_reason": {
            REASONS[r]: int((reason == r).sum()) for r in range(1, len(CRITERIA) + 1)
        },
    }


def _plot(
    metrics: dict[str, np.ndarray], ratios: np.ndarray, thresholds: dict[str, float],
    r: dict[str, float], path: Path,
) -> None:  # fmt: skip
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    panels = [(k, metrics[k], thresholds.get(k)) for k in CRITERIA]
    panels.append(("torso / box height (frames)", ratios[np.isfinite(ratios)], None))
    for ax, (name, values, thr) in zip(axes.flat, panels, strict=True):
        ax.hist(values, bins=60, color=BAR)
        ax.set_yscale("log")
        if thr is not None:
            ax.axvline(thr, color=INK, linewidth=1, linestyle="--")
            ax.text(thr, ax.get_ylim()[1], f" {thr:.4g}", color=INK, fontsize=8, va="top")
        if name.startswith("torso /"):
            for label, v in [("r_p1", r["r_p1"]), ("r_med", r["r_med"])]:
                ax.axvline(v, color=INK, linewidth=1, linestyle="--")
                ax.text(v, ax.get_ylim()[1], f" {label} {v:.3f}", color=INK, fontsize=8, va="top")
        ax.set_title(name, color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle("PoseLift training windows (T = 24, S = 6), count in log scale", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/measure_quality.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    c1 = load_config(Path(config["c1_config"]))
    dataset = load_config(Path(c1["dataset_config"]))
    config["c1"], config["dataset"] = c1, dataset
    run_dir = start_run("measure_quality", config=config, seeds={"split": c1["split"]["seed"]})

    root = Path(dataset["root"])
    train_ids, val_ids = split_train_val(
        official_split(root)["train"], c1["split"]["val_fraction"], c1["split"]["seed"]
    )
    t = c1["treatment"]
    size = (int(config["image_size"][0]), int(config["image_size"][1]))
    margin = float(config["border_margin_px"])
    train = _measure(root, train_ids, dataset, t, t["stride_train"], size, margin)

    pcts = config["percentiles"]
    percentiles = {
        k: dict(zip(map(str, pcts), np.percentile(v, pcts).tolist(), strict=True))
        for k, v in train["metrics"].items()
    }
    fits = train["boxes"]["rows_outside_image"] == 0
    thresholds = {}
    for k, rule in config["rule"].items():
        if k == "border_fraction" and not fits:
            continue  # boxes outside 1920 x 1080: criterion left out (session 2D, task 3)
        thresholds[k] = float(np.percentile(train["metrics"][k], float(rule.lstrip("p"))))
    finite = train["ratios"][np.isfinite(train["ratios"])]
    r = {"r_med": float(np.median(finite)), "r_p1": float(np.percentile(finite, 1))}

    val_s6 = _measure(root, val_ids, dataset, t, t["stride_train"], size, margin)
    val_s1 = _measure(root, val_ids, dataset, t, t["stride_eval"], size, margin)
    results = {
        "train_videos": len(train_ids),
        "val_videos": len(val_ids),
        "image_size": list(size),
        "train_boxes": train["boxes"],
        "boxes_fit_image": fits,
        "percentiles": percentiles,
        "rule": config["rule"],
        "thresholds": thresholds,
        "torso_box_ratio": {
            **r,
            "frames_with_four_torso_joints": int(len(finite)),
            "frames": int(len(train["ratios"])),
            "percentiles": dict(
                zip(map(str, pcts), np.percentile(finite, pcts).tolist(), strict=True)
            ),
        },
        "removal": {
            "train_S6": _removal(train["metrics"], thresholds),
            "val_S6": _removal(val_s6["metrics"], thresholds),
            "val_S1": _removal(val_s1["metrics"], thresholds),
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot(train["metrics"], train["ratios"], thresholds, r, run_dir / "histograms.png")
    print(json.dumps({k: results[k] for k in ["boxes_fit_image", "thresholds", "train_boxes"]}))
    print(json.dumps(r))
    print(json.dumps(results["removal"], indent=1))
    print(run_dir)


if __name__ == "__main__":
    main()
