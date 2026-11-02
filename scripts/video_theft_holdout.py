"""Person-crop theft classifier: `dev` compares operating points on the 21 annotated videos;
`final` trains on all 21 and scores the UCA-labelled videos once, with the frozen choice."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tcc.eval.alerts import alerts
from tcc.eval.events import event_hits
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run
from tcc.video.clips import (
    causal_standardise,
    clip_frame_scores,
    frame_labels,
    parse_annotations,
    remove_position_trend,
)
from tcc.video.theft_model import clip_labels, fit_linear, predict_linear, within_video_auc


def load(root: Path, name: str) -> dict[str, Any]:
    d = np.load(root / f"{name}.npz")
    return {
        "id": name,
        "x": d["x"],
        "starts": d["starts"],
        "n": int(d["n_frames"]),
        "fps": float(d["fps"]),
        "span": int(d["span"]),
    }


def transform(scores: np.ndarray, standardise: bool, min_history: int) -> np.ndarray:
    return causal_standardise(scores, min_history) if standardise else scores


def alert_result(
    videos: list[dict[str, Any]],
    scores: list[np.ndarray],
    frames: list[np.ndarray],
    threshold: float,
    base: dict[str, Any],
) -> dict[str, Any]:
    a = base["alert"]
    fired = []
    for v, s in zip(videos, scores, strict=True):
        stream = clip_frame_scores(v["starts"], s, v["span"], v["n"])
        found = alerts(
            stream,
            np.isfinite(stream),
            threshold,
            round(a["min_seconds"] * v["fps"]),
            round(a["merge_seconds"] * v["fps"]),
        )
        fired.append([x.fired for x in found])
    hours = sum(v["n"] / v["fps"] for v in videos) / 3600
    after = round(a["hit_after_seconds"] * 30)
    result = event_hits(frames, fired, after, hours)
    rng = np.random.default_rng(base["seed"])
    draws = base["null_draws"]
    null = np.asarray(
        [
            event_hits([np.roll(y, rng.integers(1, len(y))) for y in frames], fired, after, hours)[
                "alerts_hitting"
            ]
            for _ in range(draws)
        ]
    )
    return {
        **result,
        "null_hitting_mean": float(null.mean()),
        "null_hitting_p": float((1 + (null >= result["alerts_hitting"]).sum()) / (1 + draws)),
    }


def development(config: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    annotated = parse_annotations(Path(base["annotations"]).read_text(), base["category"])
    oof = np.load(config["dev_oof"])
    videos = [load(Path(config["features"]), k) for k in sorted(annotated)]
    frames = [frame_labels(v["n"], annotated[v["id"]]) for v in videos]
    out: dict[str, Any] = {}
    for name, standardise in (("raw", False), ("standardised", True)):
        scores = [transform(oof[v["id"]], standardise, config["min_history"]) for v in videos]
        pooled = np.concatenate(scores)
        pooled = pooled[np.isfinite(pooled)]
        out[name] = {}
        for q in config["quantiles"]:
            threshold = float(np.quantile(pooled, q))
            out[name][str(q)] = {
                "threshold": threshold,
                **alert_result(videos, scores, frames, threshold, base),
            }
    return out


def final(config: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    frozen = config["frozen"]
    annotated = parse_annotations(Path(base["annotations"]).read_text(), base["category"])
    root = Path(config["features"])
    train = [load(root, k) for k in sorted(annotated)]
    x = np.concatenate([v["x"] for v in train])
    y = np.concatenate(
        [
            clip_labels(
                v["starts"],
                frame_labels(v["n"], annotated[v["id"]]),
                v["span"],
                base["label_overlap"],
            )
            for v in train
        ]
    )
    model = fit_linear(x, y, base["linear"]["C"])
    labels = yaml.safe_load(Path(config["labels"]).read_text(encoding="utf-8"))["videos"]
    videos = [load(root, k) for k in sorted(labels)]
    frames = []
    for v in videos:
        spans = [(round(a * v["fps"]), round(b * v["fps"])) for a, b in labels[v["id"]]]
        frames.append(frame_labels(v["n"], spans))
    clips = [
        clip_labels(v["starts"], f, v["span"], base["label_overlap"])
        for v, f in zip(videos, frames, strict=True)
    ]
    raw = [predict_linear(model, v["x"]) for v in videos]
    position = [np.arange(len(c), dtype=np.float64) for c in clips]
    draws, seed = base["null_draws"], base["seed"]
    scores = [transform(s, frozen["standardise"], config["min_history"]) for s in raw]
    return {
        "videos": len(videos),
        "events_seconds": {k: len(v) for k, v in labels.items()},
        "within_video": within_video_auc(clips, raw, draws, seed),
        "within_video_no_trend": within_video_auc(
            clips, [remove_position_trend(s) for s in raw], draws, seed
        ),
        "position_baseline": within_video_auc(clips, position, draws, seed),
        "alerts": alert_result(videos, scores, frames, frozen["threshold"], base),
        "per_video": {
            v["id"]: within_video_auc([c], [s], draws, seed)
            for v, c, s in zip(videos, clips, raw, strict=True)
            if 0 < c.sum() < len(c)
        },
    }


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    stage = sys.argv[2]
    base = load_config(Path(config["base"]))
    run = start_run(
        f"video_theft_holdout_{stage}",
        config={**config, "base_config": base, "stage": stage},
        seeds={"split": base["seed"]},
    )
    results = development(config, base) if stage == "dev" else final(config, base)
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1) if stage == "final" else "")
    if stage == "dev":
        for name, rows in results.items():
            for q, r in rows.items():
                print(
                    name,
                    q,
                    round(r["threshold"], 3),
                    "events",
                    r["events_detected"],
                    "alerts",
                    r["alerts"],
                    "hit",
                    r["alerts_hitting"],
                    "null",
                    round(r["null_hitting_mean"], 1),
                    "p",
                    round(r["null_hitting_p"], 3),
                )
    print("run", run)


if __name__ == "__main__":
    main()
