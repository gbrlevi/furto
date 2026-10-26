"""Session 2D, task 2: what produced the 2C score peak (535.9)?

Recomputes the 2C demo window scores (C1 checkpoint, production treatment, S = 1) and,
for the highest-scoring window overall and the top-k windows of each video, records the
scale used, the torso joints above `conf_min` per frame, the mean confidence, the largest
absolute normalized coordinate and the confidence sum used by the loss (ADR-0007).
Hypotheses: (a) scale near zero; (b) loss normalized by a small confidence sum.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcc.data.skeleton import HIPS, SHOULDERS
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.pipeline.scoring import load_extracted
from tcc.preprocess.normalize import SCALE_SOURCES, normalize, window_scale
from tcc.preprocess.trajectories import split_trajectories, tracks
from tcc.preprocess.windows import raw_windows
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run

TORSO = SHOULDERS + HIPS


def _row(w: dict[str, Any], i: int, t: dict[str, Any]) -> dict[str, Any]:
    c = w["raw"][i, :, :, 2]
    first = int(w["meta"][i, 2])
    return {
        "track_id": int(w["meta"][i, 1]),
        "first_frame": first,
        "last_frame": first + t["T"] - 1,
        "score": float(w["scores"][i]),
        "score_if_loss_norm_tv": float(w["scores_tv"][i]),
        "scale": float(w["scale"][i]),
        "scale_source": SCALE_SOURCES[int(w["source"][i])],
        "torso_joints_above_conf_min_per_frame": (c[:, TORSO] >= t["conf_min"]).sum(1).tolist(),
        "mean_confidence": float(c.mean()),
        "valid_points": int((c >= t["conf_min"]).sum()),
        "max_abs_normalized_coordinate": float(np.abs(w["x"][i, :2]).max()),
        "confidence_sum_in_loss": float(w["x"][i, 2].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diagnose_peak.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    c1_run = Path(config["c1_run"])
    c1_config = json.loads((c1_run / "meta.json").read_text(encoding="utf-8"))["config"]
    t, loss_norm = c1_config["treatment"], c1_config["loss"]["loss_norm"]
    config["c1_treatment"] = t
    models = {"autoencoder": {"name": "C1 ST-GCN autoencoder", "version": c1_run.name,
                              "weights": str(c1_run / "model.pt")}}  # fmt: skip
    run_dir = start_run("diagnose_peak", config=config, seeds={}, models=models)

    ck = torch.load(c1_run / "model.pt", map_location="cpu", weights_only=True)
    m = ck["model"]
    model = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
    model.load_state_dict(ck["state"], strict=True)
    device = torch.device(config["device"])
    model.to(device).eval()

    out: dict[str, Any] = {"hypotheses": __doc__.split("Hypotheses: ")[1].strip(), "videos": {}}
    best: dict[str, Any] = {"score": -np.inf}
    for v in config["videos"]:
        vid = v["id"]
        if not isinstance(vid, str):
            raise SystemExit(f"video ids must be quoted strings, got {vid!r}")
        poses = load_extracted(Path(v["extraction_run"]) / f"{vid}_poses.npz", vid, v["camera"])
        trajs = split_trajectories(tracks(poses), t["max_gap"], t["T"])
        raw, meta = raw_windows(trajs, t["T"], t["stride_eval"])
        x = normalize(raw, t["conf_min"], t["center"]).transpose(0, 3, 1, 2)
        x = np.ascontiguousarray(x)
        scores = window_scores(model, x, device, loss_norm, config["batch_size"])
        scores_tv = window_scores(model, x, device, "tv", config["batch_size"])
        scale, source = window_scale(raw, t["conf_min"])

        w = {"raw": raw, "meta": meta, "x": x, "scores": scores, "scores_tv": scores_tv,
             "scale": scale, "source": source}  # fmt: skip
        finite = np.flatnonzero(np.isfinite(scores))
        top = finite[np.argsort(scores[finite])[::-1][: config["top_k"]]]
        sources = {name: int((source == k).sum()) for k, name in SCALE_SOURCES.items()}
        out["videos"][vid] = {
            "windows": int(len(x)),
            "scale_sources_all_windows": sources,
            "top": [_row(w, int(i), t) for i in top],
        }
        if len(top) and scores[top[0]] > best["score"]:
            best = {"video": vid, **_row(w, int(top[0]), t)}
        print(vid, len(x), "windows; top score", float(scores[top[0]]) if len(top) else None)
    out["peak"] = best
    (run_dir / "diagnosis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(run_dir)


if __name__ == "__main__":
    main()
