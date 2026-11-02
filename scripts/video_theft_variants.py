"""Linear clip classifiers trained only inside shops, with scene removal and person crops.

Negatives are the unannotated clips of the theft videos of the training folds, so the model cannot
win by recognising a shop. Every result is also given with the trend along the video removed.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from tcc.eval.alerts import alerts
from tcc.eval.events import event_hits
from tcc.eval.threshold import candidate_thresholds, threshold_for_rate
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run
from tcc.video.clips import (
    clip_frame_scores,
    frame_labels,
    parse_annotations,
    remove_position_trend,
    remove_scene,
)
from tcc.video.theft_model import clip_labels, fit_linear, predict_linear, within_video_auc


def load(path: Path, scene: bool) -> dict[str, Any]:
    d = np.load(path)
    x = d["x"]
    return {
        "id": path.stem,
        "x": remove_scene(x) if scene else x,
        "starts": d["starts"],
        "n": int(d["n_frames"]),
        "fps": float(d["fps"]),
        "span": int(d["span"]),
    }


def stream(v: dict[str, Any], scores: np.ndarray) -> np.ndarray:
    return clip_frame_scores(v["starts"], scores, v["span"], v["n"])


def shots(s: np.ndarray, threshold: float, fps: float, a: dict[str, Any]) -> list[Any]:
    return alerts(
        s, np.isfinite(s), threshold, round(a["min_seconds"] * fps), round(a["merge_seconds"] * fps)
    )


def hours(videos: list[dict[str, Any]]) -> float:
    return sum(v["n"] / v["fps"] for v in videos) / 3600


def evaluate(
    variant: dict[str, Any], base: dict[str, Any], quantiles: list[float], run: Path, name: str
) -> dict[str, Any]:
    annotated = parse_annotations(Path(base["annotations"]).read_text(), base["category"])
    root = Path(variant["features"])
    files = {p.stem: p for p in sorted(root.glob("*.npz"))}
    strong = [load(files[k], variant["scene"]) for k in sorted(annotated)]
    normal = [load(p, variant["scene"]) for k, p in files.items() if k.startswith("Normal_Videos")]
    rng = np.random.default_rng(base["seed"])
    order = rng.permutation(len(normal))
    f = base["normal_split"]
    n_train, n_thr = int(f[0] * len(normal)), int(f[1] * len(normal))
    n_t = [normal[i] for i in order[n_train : n_train + n_thr]]
    n_m = [normal[i] for i in order[n_train + n_thr :]]
    frames = [frame_labels(v["n"], annotated[v["id"]]) for v in strong]
    clips = [
        clip_labels(v["starts"], y, v["span"], base["label_overlap"])
        for v, y in zip(strong, frames, strict=True)
    ]
    a = base["alert"]
    after = round(a["hit_after_seconds"] * 30)
    groups = np.arange(len(strong))
    oof: list[np.ndarray] = [np.zeros(0)] * len(strong)
    fired: dict[int, list[list[int]]] = {r: [[] for _ in strong] for r in a["rates"]}
    false_normal = dict.fromkeys(a["rates"], 0.0)
    for tr, te in GroupKFold(base["folds"]).split(groups, groups, groups):
        x = np.concatenate([strong[i]["x"] for i in tr])
        y = np.concatenate([clips[i] for i in tr])
        model = fit_linear(x, y, base["linear"]["C"])
        for i in te:
            oof[i] = predict_linear(model, strong[i]["x"])
        op = [(s, np.isfinite(s)) for s in (stream(v, predict_linear(model, v["x"])) for v in n_t)]
        cand = candidate_thresholds(np.concatenate([s[m] for s, m in op]), 0.5, 1.0, 1000)
        for r in a["rates"]:
            th = threshold_for_rate(
                op, r, cand, round(a["min_seconds"] * 30), round(a["merge_seconds"] * 30), 30.0
            )[0]
            for i in te:
                s = stream(strong[i], oof[i])
                fired[r][i] = [al.fired for al in shots(s, th, strong[i]["fps"], a)]
            false_normal[r] += sum(
                len(shots(stream(v, predict_linear(model, v["x"])), th, v["fps"], a)) for v in n_m
            )
    draws, seed = base["null_draws"], base["seed"]

    def with_null(found: list[list[int]]) -> dict[str, Any]:
        e = event_hits(frames, found, after, hours(strong))
        null = np.asarray(
            [
                event_hits(
                    [np.roll(y, rng.integers(1, len(y))) for y in frames],
                    found,
                    after,
                    hours(strong),
                )["alerts_hitting"]
                for _ in range(draws)
            ]
        )
        return {
            **e,
            "null_hitting_mean": float(null.mean()),
            "null_hitting_p": float((1 + (null >= e["alerts_hitting"]).sum()) / (1 + draws)),
        }

    pooled = np.concatenate(oof)
    curve = {}
    for q in quantiles:
        th = float(np.quantile(pooled, q))
        curve[str(q)] = with_null(
            [
                [al.fired for al in shots(stream(v, o), th, v["fps"], a)]
                for v, o in zip(strong, oof, strict=True)
            ]
        )
    np.savez_compressed(
        run / f"oof_{name}.npz", **{v["id"]: o for v, o in zip(strong, oof, strict=True)}
    )
    return {
        "within_video": within_video_auc(clips, oof, draws, seed),
        "within_video_no_trend": within_video_auc(
            clips, [remove_position_trend(o) for o in oof], draws, seed
        ),
        "normal_thresholds": {
            str(r): {
                **with_null(fired[r]),
                "false_per_hour_normal": false_normal[r] / (base["folds"] * hours(n_m)),
            }
            for r in a["rates"]
        },
        "shop_curve_for_the_record": curve,
    }


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    base = load_config(Path(config["base"]))
    run = start_run(
        "video_theft_variants",
        config={**config, "base_config": base},
        seeds={"split": base["seed"]},
    )
    results: dict[str, Any] = {}
    for name, variant in config["variants"].items():
        results[name] = evaluate(variant, base, config["curve_quantiles"], run, name)
        print(name, json.dumps(results[name]), flush=True)
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("run", run)


if __name__ == "__main__":
    main()
