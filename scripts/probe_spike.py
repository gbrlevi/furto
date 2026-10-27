"""Session 3, D.10: is a validation-loss spike caused by its largest-loss batch?

Scores the windows of the largest-loss batch of the spike epoch with the saved checkpoint
(best epoch, which must be the epoch before the spike) and compares them with all the
training windows. A batch that is ordinary under the pre-spike weights points to an
unstable update inside the epoch, not to pathological windows. Descriptive only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tcc.data.poselift import load_video_with_real
from tcc.data.split import official_split, split_train_val
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.preprocess.normalize import SCALE_SOURCES
from tcc.preprocess.quality import SCORABLE
from tcc.preprocess.windows import treated_windows
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def _stats(v: np.ndarray, sel: np.ndarray) -> dict[str, float]:
    return {
        "all_mean": float(v.mean()), "all_median": float(np.median(v)),
        "all_p99": float(np.percentile(v, 99)), "all_max": float(v.max()),
        "batch_mean": float(v[sel].mean()), "batch_median": float(np.median(v[sel])),
        "batch_max": float(v[sel].max()),
        "share_of_windows_above_batch_max": float((v > v[sel].max()).mean()),
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/probe_spike.yaml"))
    config = load_config(parser.parse_args().config)
    run = Path(config["train_run"])
    c = json.loads((run / "meta.json").read_text(encoding="utf-8"))["config"]
    config["train_config"] = c
    out_dir = start_run("probe_spike", config=config, seeds={"split": c["split"]["seed"]},
                        models={"autoencoder": {"name": "C1", "version": run.name,
                                                "weights": str(run / "model.pt")}})  # fmt: skip
    t, d = c["treatment"], c["dataset"]
    root, size = Path(d["root"]), (int(d["image_size"][0]), int(d["image_size"][1]))
    train_ids, _ = split_train_val(
        official_split(root)["train"], c["split"]["val_fraction"], c["split"]["seed"]
    )
    xs, sources, names = [], [], []
    for vid in train_ids:  # same order as scripts/train.py, so the same window indices
        video, real = load_video_with_real(root, "train", vid, d)
        w = treated_windows(video, t, t["stride_train"], size, real)
        ok = w.reason == SCORABLE
        xs.append(w.x[ok])
        sources.append(w.scale_source[ok])
        names += [(vid, int(a), int(b)) for _, a, b in w.meta[ok].tolist()]
    x, source = np.concatenate(xs), np.concatenate(sources)
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    m = ck["model"]
    net = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
    net.load_state_dict(ck["state"], strict=True)
    device = torch.device(config["device"])
    net.to(device)
    error = window_scores(net, x, device, c["loss"]["loss_norm"], config["batch_size"])
    epoch = int(config["epoch"])
    batch = json.loads((run / "max_loss_batches.json").read_text(encoding="utf-8"))[epoch - 1]
    index = {n: i for i, n in enumerate(names)}
    sel = np.array([index[tuple(w)] for w in batch["windows"]])
    history = (run / "history.csv").read_text(encoding="utf-8").splitlines()
    results = {
        "checkpoint_epoch": int(ck["epoch"]),
        "spike_epoch": epoch,
        "batch_loss_in_training": batch["batch_loss_max"],
        "batch_windows": int(len(sel)),
        "windows": int(len(x)),
        "error_under_checkpoint": _stats(error, sel),
        "max_abs_coordinate": _stats(np.abs(x[:, :2]).reshape(len(x), -1).max(1), sel),
        "scale_source_in_batch": {
            n: int((source[sel] == k).sum()) for k, n in SCALE_SOURCES.items()
        },
        "history_rows_around_spike": history[:1] + history[epoch - 3 : epoch + 3],
    }  # fmt: skip
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1))
    print(out_dir)


if __name__ == "__main__":
    main()
