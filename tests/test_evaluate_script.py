"""End-to-end check of scripts/evaluate.py on the synthetic PoseLift (session 3, D.8).

Exercises the paths that only run with `--final` (offline metrics, bootstrap, occlusion)
on synthetic data with an untrained model, so that they do not fail for the first time on
the real test, where a crash still counts as an opening (ADR-0009).
"""

import json
from pathlib import Path

import numpy as np
import torch

from scripts import evaluate
from tcc.data.split import official_split
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.utils.frozen import sha256
from tests.fixtures.synthetic import generate, write_poselift

DATASET = {
    "keypoint_order": "yx_stored", "bbox_format": "xyxy", "nan_confidence": "interpolate",
    "image_size": [1920, 1080], "fps": 15,
}  # fmt: skip
TREATMENT = {
    "T": 24, "stride_train": 6, "stride_eval": 1, "max_gap": 8, "conf_min": 0.1,
    "center": "frame", "scale_guard": "on", "scale_ratio": {"r_med": 0.31, "r_p1": 0.18},
    "quality_gate": "on", "border_margin_px": 2,
    "quality_thresholds": {"mean_confidence": 0.2, "real_fraction": 0.6, "box_height": 10.0},
}  # fmt: skip
CONFIG = {
    "operating_points": {"main": 5, "secondary": [2, 10]},
    "alerts": {"min_alert_frames": 15, "merge_gap_frames": 30},
    "threshold_candidates": {"low": 0.5, "high": 1.0, "n": 200},
    "offline": {"frame_score": "center", "sigmas": [1, 2, 3, 4, 5, 6]},
    "events": {"after_frames": 30},
    "bootstrap": {"n_resamples": 20, "seed": 0, "level": 0.95},
    "batch_size": 256,
}  # fmt: skip


def _model_run(tmp_path: Path) -> Path:
    data = tmp_path / "poselift"
    write_poselift(data, generate(seed=2, n_train=4, n_test=3))
    run = tmp_path / "run"
    run.mkdir()
    torch.manual_seed(0)
    net = STGCNAutoencoder()
    train_ids = official_split(data)["train"]
    state = {"state": net.state_dict(), "model": {"channels": [32, 64, 64], "latent": 16,
             "k_t": 9, "dropout": 0.1}, "val_ids": train_ids[:2]}  # fmt: skip
    torch.save(state, run / "model.pt")
    config = {"treatment": TREATMENT, "dataset": DATASET | {"root": str(data)},
              "loss": {"loss_norm": "conf"}}  # fmt: skip
    (run / "meta.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    return run


def test_test_mode_metrics_bootstrap_and_occlusion_run_on_synthetic_data(tmp_path: Path) -> None:
    run = _model_run(tmp_path)
    model = evaluate.Model(run, sha256(run / "model.pt"), torch.device("cpu"))
    thresholds = evaluate._thresholds(model, CONFIG, 15.0)
    assert set(thresholds) == {"5", "2", "10"}
    test_ids = official_split(Path(model.dataset["root"]))["test"]
    videos = [evaluate._video(model, "test", vid, 256) for vid in test_ids]
    m = evaluate._metrics(videos, thresholds, CONFIG, 15.0)
    assert 0.0 <= m["offline"]["auc_roc"] <= 1.0
    assert set(m["offline"]) == {
        "auc_roc", "auc_pr_theft_average_precision", "auc_pr_normal_trapezoid", "eer",
    }  # fmt: skip
    assert m["theft_frames"] > 0 and m["theft_frames_in_filtered_windows"] is not None
    for op in m["operational"].values():
        assert {"precision", "recall", "f1", "h_prs"} <= op["frame"].keys()
        assert op["events"]["events"] > 0
    occluded = [
        evaluate._video(model, "test", vid, 256, 0.5, np.random.default_rng(i))
        for i, vid in enumerate(test_ids)
    ]
    om = evaluate._metrics(occluded, thresholds, CONFIG, 15.0)
    assert sum(om["not_scorable"]["center"].values()) >= sum(m["not_scorable"]["center"].values())
    scores = m.pop("_offline_scores")
    summary = evaluate._summary([m, m])
    assert summary["offline.auc_roc"]["std"] == 0.0
    labels = [v["labels"] for v in videos]
    b = evaluate.bootstrap_by_video(labels, [scores, scores], evaluate.auc_roc, 20, 0, 0.95)
    assert b["low"] <= m["offline"]["auc_roc"] <= b["high"]


def test_checkpoint_with_another_hash_is_refused(tmp_path: Path) -> None:
    run = _model_run(tmp_path)
    try:
        evaluate.Model(run, "0" * 64, torch.device("cpu"))
    except SystemExit as e:
        assert "sha256" in str(e)
    else:
        raise AssertionError("a checkpoint with another hash was accepted")
