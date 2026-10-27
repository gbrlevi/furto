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

from tcc.data.poselift import load_video_with_real
from tcc.data.split import official_split, split_train_val
from tcc.model.dataset import WindowDataset
from tcc.model.stgcn_ae import STGCNAutoencoder, count_parameters
from tcc.model.train import loss_spikes, train
from tcc.preprocess.normalize import SCALE_SOURCES
from tcc.preprocess.quality import REASONS, SCORABLE
from tcc.preprocess.windows import check_no_theft, treated_windows
from tcc.utils.config import load_config
from tcc.utils.run_logger import record_artifact, start_run
from tcc.utils.seed import set_seed

TRAIN_COLOR, VAL_COLOR, INK, GRID = "#2a78d6", "#eb6834", "#3d3d3a", "#e4e3df"


def _windows(
    root: Path, ids: list[str], dataset: dict, t: dict
) -> tuple[np.ndarray, dict, list[tuple[str, int, int]]]:
    """Scorable windows of the videos (ADR-0014), counts of gate reasons and scale
    sources (ADR-0015) over all windows, and (video, track, first frame) of each
    scorable window in dataset order."""
    blocks, reasons, sources, names = [], [], [], []
    size = (int(dataset["image_size"][0]), int(dataset["image_size"][1]))
    for vid in ids:
        video, real = load_video_with_real(root, "train", vid, dataset)
        w = treated_windows(video, t, t["stride_train"], size, real)
        check_no_theft(w.meta, video.frame_labels, t["T"])
        blocks.append(w.x[w.reason == SCORABLE])
        names += [(vid, int(tr), int(q)) for _, tr, q in w.meta[w.reason == SCORABLE].tolist()]
        reasons.append(w.reason)
        sources.append(w.scale_source)
    reason, source = np.concatenate(reasons), np.concatenate(sources)
    counts = {
        "windows_before_gate": int(len(reason)),
        "gate_reasons": {name: int((reason == r).sum()) for r, name in REASONS.items()},
        "scale_sources": {name: int((source == k).sum()) for k, name in SCALE_SOURCES.items()},
        "scale_sources_scorable": {
            name: int(((source == k) & (reason == SCORABLE)).sum())
            for k, name in SCALE_SOURCES.items()
        },
    }
    return np.concatenate(blocks), counts, names


def _plot(history: list[dict[str, float]], best_epoch: int, path: Path, title: str) -> None:
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
    ax.set_title(title, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_instrumentation(history: list[dict[str, float]], path: Path, title: str) -> None:
    """Gradient norm and largest batch loss per epoch (session 3, P16 item 11)."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
    series = [
        (axes[0], [("grad_norm_max", TRAIN_COLOR, "max"), ("grad_norm_mean", VAL_COLOR, "mean")],
         "gradient norm (log)"),
        (axes[1], [("batch_loss_max", TRAIN_COLOR, "largest batch"),
                   ("train_loss", VAL_COLOR, "epoch mean")], "train loss (log)"),
    ]  # fmt: skip
    for ax, lines, ylabel in series:
        for key, color, name in lines:
            ax.plot(epochs, [h[key] for h in history], color=color, linewidth=1.5, label=name)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("epoch", color=INK)
    axes[0].set_title(title, color=INK)
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
    run_dir = start_run(f"train_{args.config.stem}", config=config, seeds=seeds)

    rng = set_seed(config["train"]["seed"])
    root = Path(dataset["root"])
    train_ids, val_ids = split_train_val(
        official_split(root)["train"], config["split"]["val_fraction"], config["split"]["seed"]
    )
    t = config["treatment"]
    (x_train, counts_train, names_train), (x_val, counts_val, _) = (
        _windows(root, train_ids, dataset, t),
        _windows(root, val_ids, dataset, t),
    )

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
    record_artifact(run_dir, "model.pt", run_dir / "model.pt")  # P17 item 18
    batches = [
        {"epoch": h["epoch"], "batch_loss_max": h["batch_loss_max"],
         "windows": [names_train[i] for i in idx]}
        for h, idx in zip(result.history, result.max_loss_batches, strict=True)
    ]  # fmt: skip
    (run_dir / "max_loss_batches.json").write_text(json.dumps(batches), encoding="utf-8")
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.history[0]))
        writer.writeheader()
        writer.writerows(result.history)
    _plot(result.history, result.best_epoch, run_dir / "loss_curve.png",
          f"C1 training on PoseLift, {args.config.stem} (validation only)")  # fmt: skip
    _plot_instrumentation(result.history, run_dir / "instrumentation.png",
                          f"{args.config.stem}: gradient and batch loss per epoch")  # fmt: skip
    spike = config["train"]["spike_check"]
    spikes = loss_spikes(
        [h["val_loss"] for h in result.history],
        spike["factor"],
        spike["recover_epochs"],
        spike["recover_factor"],
    )
    seconds = [h["seconds"] for h in result.history]
    results = {
        "parameters": count_parameters(model),
        "videos": {"train": len(train_ids), "val": len(val_ids)},
        "val_ids": val_ids,
        "windows": {"train": int(len(x_train)), "val": int(len(x_val))},
        "treatment_counts": {"train": counts_train, "val": counts_val},
        "epochs_run": len(result.history),
        "epoch_limit": config["train"]["epochs"],
        "stopped_early": result.stopped_early,
        "val_loss_spikes": spikes,
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
    if not result.stopped_early:
        print("EPOCH LIMIT REACHED without early stopping: stop and report (P16, item 10)")


if __name__ == "__main__":
    main()
