"""What the C1 score reacts to, on validation windows only (the test is not read).

For each of the 5 seeds: the size of the latent against the input, the score of perturbed
copies of a fixed sample of windows (a concealment-like hand to the hip, a raised wrist, the
pose upside down, noise, shuffled joints), and the Spearman correlation of the score with
the speed and confidence of each window.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from tcc.data.poselift import load_video_with_real
from tcc.eval.controls import (
    add_noise,
    descriptors,
    hand_to_hip,
    raise_wrist,
    shuffle_joints,
    upside_down,
)
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.preprocess.quality import SCORABLE
from tcc.preprocess.windows import treated_windows
from tcc.utils.config import load_config
from tcc.utils.frozen import sha256
from tcc.utils.run_logger import start_run


def load_model(run: Path, expected: str, device: torch.device) -> tuple[Any, dict[str, Any]]:
    if sha256(run / "model.pt") != expected:
        raise SystemExit(f"{run}/model.pt does not match its frozen hash")
    c1 = json.loads((run / "meta.json").read_text(encoding="utf-8"))["config"]
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    m = ck["model"]
    net = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
    net.load_state_dict(ck["state"], strict=True)
    net.to(device).eval()
    return net, {**c1, "val_ids": list(ck["val_ids"])}


def validation_windows(c1: dict[str, Any], stride: int) -> np.ndarray:
    ds, t = c1["dataset"], {**c1["treatment"], "stride_eval": stride}
    size = (int(ds["image_size"][0]), int(ds["image_size"][1]))
    out = []
    for vid in c1["val_ids"]:
        video, real = load_video_with_real(Path(ds["root"]), "train", vid, ds)
        w = treated_windows(video, t, t["stride_eval"], size, real)
        out.append(w.x[w.reason == SCORABLE])
    return np.concatenate(out)


def latent_size(net: Any, x: np.ndarray, device: torch.device) -> int:
    seen: list[int] = []
    hook = net.to_latent.register_forward_hook(lambda _m, _i, o: seen.append(o[0].numel()))
    with torch.no_grad():
        net(torch.from_numpy(x[:1]).to(device))
    hook.remove()
    return seen[0]


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    frozen = load_config(Path(config["models"]))
    run = start_run(
        "autoencoder_controls",
        config={**config, "models_config": frozen},
        seeds={"sample": config["sample"]["seed"]},
    )
    device = torch.device(config["device"])
    results = json.loads(
        (Path(config["thresholds_from"]) / "results.json").read_text(encoding="utf-8")
    )
    thresholds = {Path(m["run"]).as_posix(): m["thresholds"] for m in results["per_model"]}
    p = config["perturbations"]
    windows: np.ndarray | None = None
    per_seed = []
    for m in frozen["models"]:
        net, c1 = load_model(Path(m["run"]), m["sha256"], device)
        x = validation_windows(c1, config["stride"])
        if windows is not None and not np.array_equal(x, windows):
            raise SystemExit("the validation windows differ between seeds")
        windows = x
        rng = np.random.default_rng(config["sample"]["seed"])
        base = x[rng.choice(len(x), min(config["sample"]["size"], len(x)), replace=False)]
        variants = {
            "original": base,
            "hand_to_hip_left": hand_to_hip(base, "left", **p["hand_to_hip"]),
            "hand_to_hip_both": hand_to_hip(
                hand_to_hip(base, "left", **p["hand_to_hip"]), "right", **p["hand_to_hip"]
            ),
            "raise_wrist_left": raise_wrist(base, "left", **p["raise_wrist"]),
            "upside_down": upside_down(base),
            "noise": add_noise(base, p["noise_sd"], rng),
            "shuffled_joints": shuffle_joints(base, rng),
        }

        def scores(w: np.ndarray, net: Any = net, c1: dict[str, Any] = c1) -> np.ndarray:
            return window_scores(net, w, device, c1["loss"]["loss_norm"], config["batch_size"])

        thr = thresholds[Path(m["run"]).as_posix()]
        reference = float(np.median(scores(base)))
        controls = {}
        for name, w in variants.items():
            s = scores(w)
            controls[name] = {
                "median": float(np.median(s)),
                "p95": float(np.quantile(s, 0.95)),
                "median_ratio": float(np.median(s) / reference),
                **{f"above_N{n}": float((s > thr[n]["threshold"]).mean()) for n in thr},
            }
        moved = hand_to_hip(base, "left", **p["hand_to_hip"])[:, :2, -1, 9] - base[:, :2, -1, 9]
        all_scores = scores(x)
        per_seed.append(
            {
                "run": m["run"],
                "validation_windows": len(x),
                "latent_values": latent_size(net, x, device),
                "input_values": int(np.prod(x.shape[1:])),
                "reconstructed_values": int(2 * np.prod(x.shape[2:])),
                "hand_to_hip_wrist_shift_torsos": {
                    "median": float(np.median(np.linalg.norm(moved, axis=1))),
                    "p95": float(np.quantile(np.linalg.norm(moved, axis=1), 0.95)),
                },
                "controls": controls,
                "spearman": {
                    k: float(spearmanr(v, all_scores).statistic) for k, v in descriptors(x).items()
                },
            }
        )
        print(m["run"], json.dumps(per_seed[-1]["spearman"]), flush=True)
    summary: dict[str, Any] = {"per_seed": per_seed}
    for name in per_seed[0]["controls"]:
        ratios = [s["controls"][name]["median_ratio"] for s in per_seed]
        summary.setdefault("median_ratio_mean_sd", {})[name] = [
            float(np.mean(ratios)),
            float(np.std(ratios, ddof=1)),
        ]
    for name in per_seed[0]["spearman"]:
        values = [s["spearman"][name] for s in per_seed]
        summary.setdefault("spearman_mean_sd", {})[name] = [
            float(np.mean(values)),
            float(np.std(values, ddof=1)),
        ]
    (run / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=1))
    print({k: per_seed[0][k] for k in ("validation_windows", "latent_values", "input_values")})
    print("run", run)


if __name__ == "__main__":
    main()
