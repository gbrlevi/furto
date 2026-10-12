"""Train the ST-GCN autoencoder on the PoseLift training partition (validation only).

No threshold and no test metric is computed here (rules 1 and 2; ADR-0009).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tcc.data.poselift import load_video
from tcc.data.split import official_split, split_train_val
from tcc.model.dataset import WindowDataset
from tcc.model.stgcn_ae import STGCNAutoencoder, count_parameters
from tcc.model.train import train
from tcc.preprocess.windows import check_no_theft, video_windows
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run
from tcc.utils.seed import set_seed

TRAIN_COLOR, VAL_COLOR, INK, GRID = "#2a78d6", "#eb6834", "#3d3d3a", "#e4e3df"


def _windows(root: Path, ids: list[str], dataset: dict, t: dict) -> np.ndarray:
    blocks = []
    for vid in ids:
        video = load_video(root, "train", vid, dataset)
        x, meta = video_windows(
            video, t["T"], t["stride_train"], t["max_gap"], t["conf_min"], t["center"]
        )
        check_no_theft(meta, video.frame_labels, t["T"])
        blocks.append(x)
    return np.concatenate(blocks)


def _plot(history: list[dict[str, float]], best_epoch: int, path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for key, color, name in [("train_loss", TRAIN_COLOR, "train"), ("val_loss", VAL_COLOR, "val")]:
        ys = [h[key] for h in history]
        ax.plot(epochs, ys, color=color, linewidth=2, label=name)
        ax.text(epochs[-1], ys[-1], f"  {name}", color=INK, fontsize=8, va="center")
    ax.axvline(best_epoch, color=INK, linewidth=0.8, linestyle="--")
    ax.text(
        best_epoch, ax.get_ylim()[1], f" best epoch {best_epoch}", color=INK, fontsize=8, va="top"
    )
    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("weighted reconstruction error (log)", color=INK)
    ax.set_title("C1 training on PoseLift (validation only)", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/c1_base.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = load_config(Path(config["dataset_config"]))
    config["dataset"] = dataset
    seeds = {"train": config["train"]["seed"], "split": config["split"]["seed"]}
    run_dir = start_run("train_c1", config=config, seeds=seeds)

    rng = set_seed(config["train"]["seed"])
    root = Path(dataset["root"])
    train_ids, val_ids = split_train_val(
        official_split(root)["train"], config["split"]["val_fraction"], config["split"]["seed"]
    )
    t = config["treatment"]
    x_train, x_val = _windows(root, train_ids, dataset, t), _windows(root, val_ids, dataset, t)

    aug = config["augmentation"]
    mask_target = config["loss"]["mask_target"]
    train_set = WindowDataset(
        x_train,
        rng,
        augment=aug["enabled"],
        p_drop_joint=aug["p_drop_joint"],
        rotation_deg=aug["rotation_deg"],
        scale=aug["scale"],
        noise=aug["noise"],
        p_mirror=aug["p_mirror"],
        mask_target=mask_target,
    )
    val_set = WindowDataset(x_val, rng)

    m = config["model"]
    model = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def log(row: dict[str, float]) -> None:
        print(
            f"epoch {int(row['epoch']):3d} | train {row['train_loss']:.5f} | "
            f"val {row['val_loss']:.5f} | lr {row['lr']:.2g} | {row['seconds']:.1f}s",
            flush=True,
        )

    generator = torch.Generator().manual_seed(config["train"]["seed"])
    result = train(
        model,
        train_set,
        val_set,
        config["train"],
        config["loss"]["loss_norm"],
        device,
        generator,
        log,
    )

    torch.save(
        {"state": result.best_state, "model": m, "epoch": result.best_epoch, "val_ids": val_ids},
        run_dir / "model.pt",
    )
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.history[0]))
        writer.writeheader()
        writer.writerows(result.history)
    _plot(result.history, result.best_epoch, run_dir / "loss_curve.png")
    seconds = [h["seconds"] for h in result.history]
    results = {
        "parameters": count_parameters(model),
        "videos": {"train": len(train_ids), "val": len(val_ids)},
        "val_ids": val_ids,
        "windows": {"train": int(len(x_train)), "val": int(len(x_val))},
        "epochs_run": len(result.history),
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "seconds_per_epoch": {"mean": float(np.mean(seconds)), "max": float(np.max(seconds))},
        "device": str(device),
        "vram_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "vram_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(run_dir)
    print(json.dumps({k: v for k, v in results.items() if k != "val_ids"}, indent=2))


if __name__ == "__main__":
    main()
