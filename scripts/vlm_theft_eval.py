"""Evaluation of the zero-shot clip scores.

Usage: vlm_theft_eval.py CONFIG SET PROMPT [PROMPT ...], SET in dev, holdout.
Within-video AUC against a rolled-label null, the position-in-video baseline, and alerts with
thresholds for N alerts per hour taken from the threshold part of the normal videos (false
alerts counted on the measure part), when those videos have been scored.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
)
from tcc.video.theft_model import clip_labels, within_video_auc


def load(path: Path) -> dict[str, Any]:
    d = np.load(path)
    return {
        "id": path.stem,
        "scores": d["scores"],
        "starts": d["starts"],
        "span": int(d["span"]),
        "n": int(d["n_frames"]),
        "fps": float(d["fps"]),
    }


def stream(v: dict[str, Any]) -> np.ndarray:
    return clip_frame_scores(v["starts"], v["scores"], v["span"], v["n"])


def fired(v: dict[str, Any], threshold: float, a: dict[str, Any]) -> list[int]:
    s = stream(v)
    found = alerts(
        s,
        np.isfinite(s),
        threshold,
        round(a["min_seconds"] * v["fps"]),
        round(a["merge_seconds"] * v["fps"]),
    )
    return [x.fired for x in found]


def hours(videos: list[dict[str, Any]]) -> float:
    return sum(v["n"] / v["fps"] for v in videos) / 3600


def with_null(
    frames: list[np.ndarray], found: list[list[int]], base: dict[str, Any], h: float
) -> dict[str, Any]:
    after = round(base["alert"]["hit_after_seconds"] * 30)
    result = event_hits(frames, found, after, h)
    rng = np.random.default_rng(base["seed"])
    null = np.asarray(
        [
            event_hits([np.roll(y, rng.integers(1, len(y))) for y in frames], found, after, h)[
                "alerts_hitting"
            ]
            for _ in range(base["null_draws"])
        ]
    )
    return {
        **result,
        "null_hitting_mean": float(null.mean()),
        "null_hitting_p": float((1 + (null >= result["alerts_hitting"]).sum()) / (1 + len(null))),
    }


def labelled(config: dict[str, Any], base: dict[str, Any], which: str) -> dict[str, list]:
    if which == "dev":
        spans = parse_annotations(Path(base["annotations"]).read_text(), base["category"])
        return {k: [("frames", a, b) for a, b in v] for k, v in spans.items()}
    labels = yaml.safe_load(Path(config["holdout_labels"]).read_text(encoding="utf-8"))
    return {k: [("seconds", a, b) for a, b in v] for k, v in labels["videos"].items()}


def normal_parts(config: dict[str, Any], base: dict[str, Any]) -> tuple[list[str], list[str]]:
    normal = sorted(p.stem for p in Path(config["normal_videos"]).glob("*.mp4"))
    order = np.random.default_rng(base["seed"]).permutation(len(normal))
    f = base["normal_split"]
    n_train, n_thr = int(f[0] * len(normal)), int(f[1] * len(normal))
    return (
        [normal[i] for i in order[n_train : n_train + n_thr]],
        [normal[i] for i in order[n_train + n_thr :]],
    )


def evaluate(config: dict[str, Any], base: dict[str, Any], which: str, prompt: str) -> dict:
    root = Path(config["out"]) / prompt
    spans = labelled(config, base, which)
    videos = [load(root / f"{k}.npz") for k in sorted(spans)]
    frames = []
    for v in videos:
        rows = spans[v["id"]]
        f = [
            (a, b) if unit == "frames" else (round(a * v["fps"]), round(b * v["fps"]))
            for unit, a, b in rows
        ]
        frames.append(frame_labels(v["n"], f))
    clips = [
        clip_labels(v["starts"], y, v["span"], base["label_overlap"])
        for v, y in zip(videos, frames, strict=True)
    ]
    draws, seed = base["null_draws"], base["seed"]
    scores = [v["scores"] for v in videos]
    out: dict[str, Any] = {
        "videos": len(videos),
        "events": int(sum(len(r) for r in spans.values())),
        "within_video": within_video_auc(clips, scores, draws, seed),
        "within_video_no_trend": within_video_auc(
            clips, [remove_position_trend(s) for s in scores], draws, seed
        ),
        "position_baseline": within_video_auc(
            clips, [np.arange(len(c), dtype=np.float64) for c in clips], draws, seed
        ),
    }
    thr_ids, measure_ids = normal_parts(config, base)
    if all((root / f"{k}.npz").exists() for k in thr_ids + measure_ids):
        a = base["alert"]
        thr_videos = [load(root / f"{k}.npz") for k in thr_ids]
        measure = [load(root / f"{k}.npz") for k in measure_ids]
        streams = [(s, np.isfinite(s)) for s in map(stream, thr_videos)]
        cand = candidate_thresholds(np.concatenate([s[m] for s, m in streams]), 0.5, 1.0, 1000)
        out["alerts"] = {}
        for rate in a["rates"]:
            th = threshold_for_rate(
                streams,
                rate,
                cand,
                round(a["min_seconds"] * 30),
                round(a["merge_seconds"] * 30),
                30.0,
            )[0]
            out["alerts"][str(rate)] = {
                "threshold": th,
                **with_null(frames, [fired(v, th, a) for v in videos], base, hours(videos)),
                "false_per_hour_normal": sum(len(fired(v, th, a)) for v in measure)
                / hours(measure),
            }
    return out


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    which, prompts = sys.argv[2], sys.argv[3:]
    base = load_config(Path(config["base"]))
    run = start_run(
        f"vlm_theft_eval_{which}",
        config={**config, "set": which, "prompts_run": prompts, "base_config": base},
        seeds={"null": base["seed"]},
    )
    results = {p: evaluate(config, base, which, p) for p in prompts}
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1))
    print("run", run)


if __name__ == "__main__":
    main()
