"""RGB theft detection on UCF-Crime Shoplifting: cross-validation by video against chance.

Writes results.json and per-model scores in the run directory.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from tcc.eval.alerts import alerts
from tcc.eval.events import event_hits
from tcc.eval.threshold import candidate_thresholds, threshold_for_rate
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run
from tcc.video.clips import clip_frame_scores, frame_labels, parse_annotations
from tcc.video.theft_model import (
    clip_labels,
    fit_linear,
    fit_mil,
    predict_linear,
    predict_mil,
    within_video_auc,
)

DEVICE = "cuda"


def load(path: Path) -> dict[str, Any]:
    d = np.load(path)
    return {
        "id": path.stem,
        "x": d["x"],
        "starts": d["starts"],
        "n": int(d["n_frames"]),
        "fps": float(d["fps"]),
        "span": int(d["span"]),
    }


def stream(v: dict[str, Any], clip_scores: np.ndarray) -> np.ndarray:
    return clip_frame_scores(v["starts"], clip_scores, v["span"], v["n"])


def fired(s: np.ndarray, threshold: float, fps: float, c: dict[str, Any]) -> list[Any]:
    return alerts(
        s,
        np.isfinite(s),
        threshold,
        round(c["min_seconds"] * fps),
        round(c["merge_seconds"] * fps),
    )


def hours(videos: list[dict[str, Any]]) -> float:
    return sum(v["n"] / v["fps"] for v in videos) / 3600


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    run = start_run("video_theft_eval", config=config, seeds={"split": config["seed"]})
    root = Path(config["features"])
    annotated = parse_annotations(Path(config["annotations"]).read_text(), config["category"])
    files = {p.stem: p for p in sorted(root.glob("*.npz"))}
    strong = [load(files[k]) for k in sorted(annotated) if k in files]
    weak = [
        load(p) for k, p in files.items() if k.startswith(config["category"]) and k not in annotated
    ]
    normal = [load(p) for k, p in files.items() if k.startswith("Normal_Videos")]
    synthetic = [load(p) for k, p in files.items() if k.startswith(config["synthetic_prefix"])]
    meva = [load(p) for k, p in files.items() if k.startswith("2018-")]
    a = config["alert"]
    print(len(strong), "annotated", len(weak), "weak", len(normal), "normal", flush=True)

    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(len(normal))
    f = config["normal_split"]
    n_train, n_thr = int(f[0] * len(normal)), int(f[1] * len(normal))
    n_a = [normal[i] for i in order[:n_train]]
    n_t = [normal[i] for i in order[n_train : n_train + n_thr]]
    n_m = [normal[i] for i in order[n_train + n_thr :]]
    print("normal hours train, threshold, measure", hours(n_a), hours(n_t), hours(n_m), flush=True)

    frames = [frame_labels(v["n"], annotated[v["id"]]) for v in strong]
    clips = [
        clip_labels(v["starts"], y, v["span"], config["label_overlap"])
        for v, y in zip(strong, frames, strict=True)
    ]
    theft_h = hours(strong)
    position = [np.arange(len(y), dtype=np.float64) for y in clips]
    results: dict[str, Any] = {
        "videos": {"annotated": len(strong), "weak": len(weak), "normal": len(normal)},
        "hours": {
            "annotated": theft_h,
            "normal_train": hours(n_a),
            "normal_threshold": hours(n_t),
            "normal_measure": hours(n_m),
        },
        "theft_clips": int(sum(y.sum() for y in clips)),
        "clips": int(sum(len(y) for y in clips)),
        "position_baseline": within_video_auc(
            clips, position, config["null_draws"], config["seed"]
        ),
    }
    print("position", results["position_baseline"], flush=True)

    neg_x = np.concatenate([v["x"] for v in n_a])
    groups = np.arange(len(strong))
    folds = list(GroupKFold(config["folds"]).split(groups, groups, groups))

    def linear(tr: np.ndarray, _: int) -> Callable[[np.ndarray], np.ndarray]:
        x = np.concatenate([strong[i]["x"] for i in tr] + [neg_x])
        y = np.concatenate([clips[i] for i in tr] + [np.zeros(len(neg_x), np.int64)])
        model = fit_linear(x, y, config["linear"]["C"])
        return lambda q: predict_linear(model, q)

    def mil(tr: np.ndarray, k: int) -> Callable[[np.ndarray], np.ndarray]:
        x = np.concatenate([strong[i]["x"] for i in tr] + [neg_x])
        y = np.concatenate([clips[i] for i in tr] + [np.zeros(len(neg_x), np.int64)])
        net = fit_mil(x, y, [v["x"] for v in weak], config["mil"], k, DEVICE)
        return lambda q: predict_mil(net, q, DEVICE)

    for name, builder in {"linear": linear, "mil": mil}.items():
        oof: list[np.ndarray] = [np.zeros(0)] * len(strong)
        events_by_rate: dict[int, list[list[int]]] = {r: [[] for _ in strong] for r in a["rates"]}
        false_measure = dict.fromkeys(a["rates"], 0.0)
        meva_alerts = dict.fromkeys(a["rates"], 0.0)
        synth_rows: dict[str, list[float]] = {}
        pooled_normal: list[np.ndarray] = []
        for k, (tr, te) in enumerate(folds):
            score = builder(tr, k)
            op = [(s, np.isfinite(s)) for s in (stream(v, score(v["x"])) for v in n_t)]
            sample = np.concatenate([s[m] for s, m in op])
            cand = candidate_thresholds(sample, 0.5, 1.0, 1000)
            thresholds = {
                r: threshold_for_rate(
                    op,
                    r,
                    cand,
                    round(a["min_seconds"] * 30),
                    round(a["merge_seconds"] * 30),
                    30.0,
                )[0]
                for r in a["rates"]
            }
            pooled_normal += [score(v["x"]) for v in n_m]
            for i in te:
                v = strong[i]
                oof[i] = score(v["x"])
                for r, th in thresholds.items():
                    s = stream(v, oof[i])
                    events_by_rate[r][i] = [al.fired for al in fired(s, th, v["fps"], a)]
            for r, th in thresholds.items():
                false_measure[r] += sum(
                    len(fired(stream(v, score(v["x"])), th, v["fps"], a)) for v in n_m
                )
                meva_alerts[r] += sum(
                    len(fired(stream(v, score(v["x"])), th, v["fps"], a)) for v in meva
                )
            for v in synthetic:
                synth_rows.setdefault(v["id"], []).append(float(score(v["x"]).max()))

        out: dict[str, Any] = {
            "within_video": within_video_auc(clips, oof, config["null_draws"], config["seed"]),
            "pooled_in_theft_videos": float(
                roc_auc_score(np.concatenate(clips), np.concatenate(oof))
            ),
            "theft_vs_normal_video": float(
                roc_auc_score(
                    np.concatenate(
                        [np.concatenate(clips)[np.concatenate(clips) == 1]]
                        + [np.zeros(sum(map(len, pooled_normal)), np.int64)]
                    ),
                    np.concatenate(
                        [np.concatenate(oof)[np.concatenate(clips) == 1], *pooled_normal]
                    ),
                )
            ),
            "operating": {},
            "synthetic_max_score_mean_over_folds": {
                k: float(np.mean(v)) for k, v in synth_rows.items()
            },
        }
        for r in a["rates"]:
            shots = events_by_rate[r]
            after = round(a["hit_after_seconds"] * 30)
            e = event_hits(frames, shots, after, theft_h)
            null_events = []
            for _ in range(config["null_draws"]):
                shifted = [np.roll(y, rng.integers(1, len(y))) for y in frames]
                null_events.append(event_hits(shifted, shots, after, theft_h)["alerts_hitting"])
            null = np.asarray(null_events)
            out["operating"][str(r)] = {
                **e,
                "null_alerts_hitting_mean": float(null.mean()),
                "null_alerts_hitting_p": float(
                    (1 + (null >= e["alerts_hitting"]).sum()) / (1 + len(null))
                ),
                "false_alerts_per_hour_measure_normal": false_measure[r]
                / (config["folds"] * hours(n_m)),
                "alerts_per_hour_meva": meva_alerts[r] / (config["folds"] * max(hours(meva), 1e-9)),
            }
        results[name] = out
        print(name, json.dumps(out, indent=1), flush=True)
        np.savez_compressed(
            run / f"oof_{name}.npz", **{v["id"]: o for v, o in zip(strong, oof, strict=True)}
        )
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("run", run)


if __name__ == "__main__":
    main()
