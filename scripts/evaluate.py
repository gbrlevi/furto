"""Evaluate trained C1 models (session 3, D.8; ADR-0008, ADR-0009, ADR-0014).

By default only the validation videos of each checkpoint are evaluated. `--final` opens
the official PoseLift test and requires a committed configuration in `configs/frozen/`,
the committed `PREDECLARED.md` and a clean tree; the opening is logged in
`runs/test_evaluations.csv` before any test video is read.

Per model (seed): thresholds for each N from its own validation, operational stream
(causal last frame, quality gate, alert logic); offline STG-NF protocol (centre, most
normal fill, sigma 1..6) for AUC-ROC, AUC-PR and EER; operational stream for frame
precision, recall, F1, H-PRS, events and alerts per hour. Across seeds: mean, standard
deviation and the stratified bootstrap interval of the mean AUC-ROC.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcc.data.poselift import load_video_with_real
from tcc.data.split import official_split
from tcc.eval.alerts import alerts
from tcc.eval.bootstrap import bootstrap_by_video
from tcc.eval.events import event_hits
from tcc.eval.frame_scores import (
    filtered_coverage,
    frame_reasons,
    most_normal,
    post_process,
    window_frame_scores,
)
from tcc.eval.metrics import auc_pr, auc_roc, eer, threshold_metrics
from tcc.eval.threshold import alerts_per_hour, candidate_thresholds, threshold_for_rate
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.preprocess.quality import REASONS, SCORABLE
from tcc.preprocess.windows import treated_windows
from tcc.utils.config import load_config
from tcc.utils.frozen import check_frozen, log_test_opening, sha256
from tcc.utils.run_logger import start_run

TEST_LOG = Path("runs") / "test_evaluations.csv"


class Model:
    """A trained checkpoint with the treatment and dataset of its training run."""

    def __init__(self, run: Path, expected_sha256: str, device: torch.device) -> None:
        actual = sha256(run / "model.pt")
        if actual != expected_sha256:
            raise SystemExit(f"{run}/model.pt sha256 {actual} != frozen {expected_sha256}")
        c1 = json.loads((run / "meta.json").read_text(encoding="utf-8"))["config"]
        self.run, self.t, self.dataset = run, c1["treatment"], c1["dataset"]
        self.loss_norm = c1["loss"]["loss_norm"]
        ck = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        m = ck["model"]
        self.net = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
        self.net.load_state_dict(ck["state"], strict=True)
        self.net.to(device).eval()
        self.device, self.val_ids = device, list(ck["val_ids"])


def _video(
    model: Model, split: str, vid: str, batch: int, drop: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Frame streams of one video: offline (centre) raw scores, operational (last frame)
    scores, reasons of not-scorable frames under both attributions, filtered coverage."""
    t = model.t
    root = Path(model.dataset["root"])
    size = (int(model.dataset["image_size"][0]), int(model.dataset["image_size"][1]))
    video, real = load_video_with_real(root, split, vid, model.dataset)
    w = treated_windows(video, t, t["stride_eval"], size, real, drop_fraction=drop, rng=rng)
    scores = np.full(len(w.meta), np.nan)
    ok = w.reason == SCORABLE
    scores[ok] = window_scores(model.net, w.x[ok], model.device, model.loss_norm, batch)
    n, T = video.n_frames, t["T"]
    labels = (
        np.zeros(n, np.int64) if video.frame_labels is None else video.frame_labels.astype(np.int64)
    )
    return {
        "id": vid,
        "labels": labels,
        "center_raw": window_frame_scores(w.meta, scores, T, n, "center"),
        "last": window_frame_scores(w.meta, scores, T, n, "last"),
        "reasons_center": frame_reasons(w.meta, w.reason, T, n, "center"),
        "reasons_last": frame_reasons(w.meta, w.reason, T, n, "last"),
        "filtered": filtered_coverage(w.meta, w.reason, T, n),
    }


def _operational(v: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return v["last"], np.isfinite(v["last"])


def _thresholds(model: Model, config: dict[str, Any], fps: float) -> dict[str, Any]:
    """Threshold for each N on the model's own validation (P17 item 13)."""
    val = [_video(model, "train", vid, config["batch_size"]) for vid in model.val_ids]
    streams = [_operational(v) for v in val]
    sample = np.concatenate([s[m] for s, m in streams])
    c = config["threshold_candidates"]
    candidates = candidate_thresholds(sample, c["low"], c["high"], c["n"])
    a = config["alerts"]
    out = {}
    for n in [config["operating_points"]["main"], *config["operating_points"]["secondary"]]:
        thr, reached = threshold_for_rate(
            streams, n, candidates, a["min_alert_frames"], a["merge_gap_frames"], fps
        )
        out[str(n)] = {
            "threshold": thr,
            "reached": reached,
            "validation_alerts_per_hour": alerts_per_hour(
                streams, thr, a["min_alert_frames"], a["merge_gap_frames"], fps
            ),
        }
    return out


def _metrics(
    videos: list[dict[str, Any]], thresholds: dict[str, Any], config: dict[str, Any], fps: float
) -> dict[str, Any]:
    """Offline and operational metrics of one model on one set of videos."""
    labels = [v["labels"] for v in videos]
    y = np.concatenate(labels)
    raw = [v["center_raw"] for v in videos]
    offline = post_process(raw, most_normal(raw), config["offline"]["sigmas"])
    s = np.concatenate(offline)
    frames = len(y)
    hours = frames / fps / 3600
    out: dict[str, Any] = {"frames": frames, "theft_frames": int(y.sum()), "hours": hours}
    if 0 < y.sum() < len(y):
        out["offline"] = {
            "auc_roc": auc_roc(y, s),
            "auc_pr_theft_average_precision": auc_pr(y, s, "theft", "average_precision"),
            "auc_pr_normal_trapezoid": auc_pr(y, s, "normal", "trapezoid"),
            "eer": eer(y, s),
        }
    a = config["alerts"]
    last = np.concatenate([v["last"] for v in videos])
    operational = {}
    for n, th in thresholds.items():
        thr = th["threshold"]
        found = [
            alerts(*_operational(v), thr, a["min_alert_frames"], a["merge_gap_frames"])
            for v in videos
        ]
        operational[n] = {
            "threshold": thr,
            "frame": threshold_metrics(y, np.where(np.isfinite(last), last, -np.inf), thr),
            "events": event_hits(
                labels,
                [[al.start for al in f] for f in found],
                config["events"]["after_frames"],
                hours,
            ),  # fmt: skip
            "alerts_per_hour": sum(len(f) for f in found) / hours,
        }
    out["operational"] = operational
    theft = y == 1
    out["not_scorable"] = {
        mode: {
            REASONS[r]: float((np.concatenate([v[f"reasons_{mode}"] for v in videos]) == r).mean())
            for r in REASONS
            if r != SCORABLE
        }
        for mode in ("center", "last")
    }
    filtered = np.concatenate([v["filtered"] for v in videos])
    out["theft_frames_in_filtered_windows"] = float(filtered[theft].mean()) if theft.any() else None
    return out | {"_offline_scores": offline}


def _summary(per_model: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sample standard deviation over seeds of every scalar metric."""

    def leaves(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                out |= leaves(v, f"{prefix}{k}.")
            elif isinstance(v, int | float) and not isinstance(v, bool):
                out[f"{prefix}{k}"] = float(v)
        return out

    flat = [leaves(m) for m in per_model]
    keys = [k for k in flat[0] if all(k in f for f in flat)]
    return {
        k: {
            "mean": float(np.mean([f[k] for f in flat])),
            "std": float(np.std([f[k] for f in flat], ddof=1)) if len(flat) > 1 else None,
        }
        for k in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--final", action="store_true", help="open the official test (ADR-0009)")
    args = parser.parse_args()
    frozen = check_frozen(args.config) if args.final else None
    config = load_config(args.config)
    split = "test" if args.final else "validation"
    models_meta = {
        f"autoencoder_{i}": {"name": "C1 ST-GCN autoencoder", "version": Path(m["run"]).name,
                             "weights": str(Path(m["run"]) / "model.pt")}
        for i, m in enumerate(config["models"])
    }  # fmt: skip
    config["frozen"] = frozen
    run_dir = start_run(
        f"evaluate_{config['name']}_{split}", config=config, seeds={}, models=models_meta
    )
    device = torch.device(config["device"])
    models = [Model(Path(m["run"]), m["sha256"], device) for m in config["models"]]
    if frozen is not None:
        opening = log_test_opening(TEST_LOG, run_dir, frozen)
        print(f"TEST OPENING {opening['opening_overall']} (this config: "
              f"{opening['opening_of_this_config']})")  # fmt: skip
    fps = float(models[0].dataset["fps"])
    test_ids = official_split(Path(models[0].dataset["root"]))["test"] if args.final else None
    batch = config["batch_size"]

    per_model, offline_scores, labels, results = [], [], None, {}
    for k, model in enumerate(models):
        thresholds = _thresholds(model, config, fps)
        if test_ids is None:
            videos = [_video(model, "train", vid, batch) for vid in model.val_ids]
        else:
            videos = [_video(model, "test", vid, batch) for vid in test_ids]
        m = _metrics(videos, thresholds, config, fps)
        offline_scores.append(m.pop("_offline_scores"))
        labels = [v["labels"] for v in videos]
        per_model.append({"run": str(model.run), "thresholds": thresholds, **m})
        np.savez_compressed(
            run_dir / f"frames_model{k}.npz",
            ids=np.array([v["id"] for v in videos]),
            **{f"{v['id']}__{key}": v[key] for v in videos
               for key in ("labels", "last", "reasons_center", "reasons_last", "filtered")},
            **{f"{v['id']}__offline": s for v, s in zip(videos, offline_scores[-1], strict=True)},
        )  # fmt: skip

        occlusion = config.get("occlusion")
        if occlusion and test_ids is not None:
            per_model[-1]["occlusion"] = {}
            for frac in occlusion["fractions"]:
                occ = [
                    _video(model, "test", vid, batch, frac,
                           np.random.default_rng([occlusion["seed"], round(frac * 1000), i]))
                    for i, vid in enumerate(test_ids)
                ]  # fmt: skip
                om = _metrics(occ, thresholds, config, fps)
                om.pop("_offline_scores")
                per_model[-1]["occlusion"][str(frac)] = om

    results["split"] = split
    results["per_model"] = per_model
    results["across_seeds"] = _summary(per_model)
    assert labels is not None
    if any((y == 1).any() for y in labels):
        b = config["bootstrap"]
        results["bootstrap_auc_roc"] = bootstrap_by_video(
            labels, offline_scores, auc_roc, b["n_resamples"], b["seed"], b["level"]
        )
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "per_model"}, indent=1))
    print(run_dir)


if __name__ == "__main__":
    main()
