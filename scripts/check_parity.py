"""Session 3B, task 5: our extracted poses against the PoseLift training partition.

Main environment, no GPU. Both sides in the internal convention (x, y, confidence; PoseLift
read as in ADR-0010 and ADR-0011). Compares the confidence of each joint (PoseLift: real
detections only, the rows with authors' NaN confidence are their interpolation), the
trajectory lengths, the bone proportions over the torso and the box convention: box height
over the vertical extent of the confident joints of people with nose and ankles visible.
Only measures and records; no conversion is applied (any conversion goes by ADR).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tcc.data.parity import EDGES, bone_ratios, box_extent_ratio, joint_confidence
from tcc.data.poselift import load_video_with_real
from tcc.data.schema import VideoPoses
from tcc.data.skeleton import JOINT_NAMES
from tcc.data.split import official_split
from tcc.data.stats import track_lengths
from tcc.pipeline.scoring import load_extracted
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run

COLORS = {"poselift": "#2a78d6", "meva": "#eb6834", "ucf": "#1baf7a"}
INK, GRID = "#3d3d3a", "#e4e3df"


def _quantiles(v: np.ndarray) -> dict[str, Any]:
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "p25": float(p25), "median": float(p50), "p75": float(p75)}


def _group(videos: list[VideoPoses], real: list[np.ndarray] | None, conf_min: float,
           min_length: int) -> dict[str, Any]:  # fmt: skip
    kp = np.concatenate([v.keypoints for v in videos])
    boxes = np.concatenate([v.bbox for v in videos])
    conf_rows = kp if real is None else kp[np.concatenate(real)]
    lengths = track_lengths(videos)["observed"]
    bones = bone_ratios(kp, conf_min)
    return {
        "videos": len(videos),
        "rows": int(len(kp)),
        "confidence_rows": int(len(conf_rows)),
        "joint_confidence": {JOINT_NAMES[j]: s for j, s in joint_confidence(conf_rows).items()},
        "track_length_observed": {
            **_quantiles(lengths.astype(np.float64)),
            "tracks": int(len(lengths)),
            f"fraction_below_{min_length}": float((lengths < min_length).mean()),
        },
        "bone_over_torso": {
            f"{JOINT_NAMES[a]}-{JOINT_NAMES[b]}": _quantiles(bones[:, i])
            for i, (a, b) in enumerate(EDGES)
        },
        "box_height_over_keypoint_extent": _quantiles(box_extent_ratio(boxes, kp, conf_min)),
    }


def _plot(results: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(JOINT_NAMES))
    bone_names = [f"{JOINT_NAMES[a]}-{JOINT_NAMES[b]}" for a, b in EDGES]
    for name, g in results["groups"].items():
        conf = [g["joint_confidence"][j]["median"] for j in JOINT_NAMES]
        axes[0].plot(x, conf, marker="o", markersize=4, linewidth=2, color=COLORS[name], label=name)
        med = [g["bone_over_torso"][b].get("median", np.nan) for b in bone_names]
        axes[1].plot(np.arange(len(bone_names)), med, marker="o", markersize=4, linewidth=2,
                     color=COLORS[name], label=name)  # fmt: skip
    axes[0].set_xticks(x, JOINT_NAMES, rotation=70, fontsize=7)
    axes[0].set_ylabel("median confidence", color=INK)
    axes[0].set_title("Confidence per joint", color=INK, fontsize=10)
    axes[1].set_xticks(np.arange(len(bone_names)), bone_names, rotation=70, fontsize=6)
    axes[1].set_ylabel("median bone length / torso", color=INK)
    axes[1].set_title("Bone proportions", color=INK, fontsize=10)
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/check_parity.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    c1 = load_config(Path(config["c1_config"]))
    dataset = load_config(Path(c1["dataset_config"]))
    config["dataset"] = dataset
    conf_min, t = float(c1["treatment"]["conf_min"]), int(c1["treatment"]["T"])
    run_dir = start_run("check_parity", config=config, seeds={})

    root = Path(dataset["root"])
    poselift, real = [], []
    for vid in official_split(root)["train"]:
        video, r = load_video_with_real(root, "train", vid, dataset)
        poselift.append(video)
        real.append(r)
    groups = {"poselift": _group(poselift, real, conf_min, t)}
    run = Path(config["ours_run"])
    for name, clips in config["ours"].items():
        ours = [
            load_extracted(run / f"{c['id']}_{config['ours_variant']}.npz", c["id"], c["camera"])
            for c in clips
        ]
        groups[name] = _group(ours, None, conf_min, t)
    results = {"conf_min": conf_min, "T": t, "groups": groups}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot(results, run_dir / "parity.png")
    for name, g in groups.items():
        keys = ["rows", "track_length_observed", "box_height_over_keypoint_extent"]
        print(name, json.dumps({k: g[k] for k in keys}))
    print(run_dir)


if __name__ == "__main__":
    main()
