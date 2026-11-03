"""The item pipeline on UCF-Crime: theft videos (officially annotated and UCA held out) and
normal videos, with hand-drawn and automatic zones.

Usage: ucf_items_eval.py CONFIG. Nothing is fitted on these videos: the hold threshold comes
from MERL, the zones from configs/zones/ucf_manual.yaml or scripts/auto_zones.py. A concealment
alert fires at the frame an item is let go on the body (main) or, in the variant, only when that
item's hold began near a shelf contact; alerts of one video within `merge_seconds` of the
previous one are merged. Per zone source and variant: event recall, alert precision and false
alerts per hour on the theft videos against a rolled-label null, and alerts per hour on the
normal videos. Payments are unknown on UCF-Crime: unpaid exits are not scored.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from shop_items import hold_of, load_video, polygons, run_pipeline

from tcc.eval.events import event_hits
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run
from tcc.video.clips import frame_labels, parse_annotations

VARIANTS = {"main": "alerts", "after_contact": "alerts_after_contact"}


def theft_spans(c: dict[str, Any]) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Theft intervals in seconds per set."""
    s = c["sets"]
    rows = parse_annotations(Path(s["official"]).read_text(), "Shoplifting")
    official = {
        k: [(a / s["source_fps"], b / s["source_fps"]) for a, b in v] for k, v in rows.items()
    }
    held = yaml.safe_load(Path(s["holdout"]).read_text(encoding="utf-8"))["videos"]
    return {"official": official, "holdout": {k: [tuple(x) for x in v] for k, v in held.items()}}


def merged(frames: list[int], gap: int) -> list[int]:
    out: list[int] = []
    for f in sorted(frames):
        if not out or f - out[-1] > gap:
            out.append(f)
    return out


def zone_mask(zone: Any, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for poly in polygons(zone):
        cv2.fillPoly(m, [np.int32(poly)], 1)
    return m.astype(bool)


def agreement(manual: dict[str, Any], auto: dict[str, Any], kind: str) -> dict[str, Any]:
    both, ious = 0, []
    only = {"manual": 0, "auto": 0}
    for vid in manual:
        a, b = zone_mask(manual[vid][kind], (240, 320)), zone_mask(auto[vid][kind], (240, 320))
        if a.any() and b.any():
            both += 1
            ious.append(float((a & b).sum() / (a | b).sum()))
        elif a.any():
            only["manual"] += 1
        elif b.any():
            only["auto"] += 1
    return {"both": both, "only_manual": only["manual"], "only_auto": only["auto"],
            "iou_median": float(np.median(ious)) if ious else None}  # fmt: skip


def score_set(
    events: dict[str, dict[str, Any]], spans: dict[str, list[tuple[float, float]]], key: str,
    c: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    fps, gap = c["fps"], round(c["merge_seconds"] * c["fps"])
    ids = sorted(spans)
    labels = [
        frame_labels(events[v]["n_frames"],
                     [(math.floor(a * fps), math.ceil(b * fps)) for a, b in spans[v]])
        for v in ids
    ]  # fmt: skip
    fired = [merged([a["frame"] for a in events[v][key]], gap) for v in ids]
    hours = sum(events[v]["n_frames"] for v in ids) / fps / 3600
    after = round(c["hit_after_seconds"] * fps)
    result = event_hits(labels, fired, after, hours)
    rng = np.random.default_rng(c["seed"])
    null = np.asarray([
        event_hits([np.roll(y, rng.integers(1, len(y))) for y in labels], fired, after, hours)[
            "alerts_hitting"
        ]
        for _ in range(c["null_draws"])
    ])  # fmt: skip
    return {
        "videos": len(ids),
        "hours": hours,
        **result,
        "alerts_per_hour": result["alerts"] / hours,
        "null_hitting_mean": float(null.mean()),
        "null_hitting_p": float((1 + (null >= result["alerts_hitting"]).sum()) / (1 + len(null))),
        "per_video": {
            v: {"events": len(spans[v]), "alerts": f} for v, f in zip(ids, fired, strict=True)
        },
    }


def main() -> None:
    c = load_config(Path(sys.argv[1]))
    run = start_run(c["name"], config=c, seeds={"null": c["seed"]})
    spans = theft_spans(c)
    normal = [v["id"] for v in load_config(Path(c["videos_from"]))["videos"]
              if v["camera"] == "ucf_normal"]  # fmt: skip
    ids = sorted(spans["official"]) + sorted(spans["holdout"]) + normal
    zones = {k: dict(load_config(Path(p))) for k, p in c["zones"].items()}
    results: dict[str, Any] = {
        "zone_agreement": {
            k: agreement(zones["manual"], zones["auto"], k) for k in ("shelf", "checkout")
        }  # fmt: skip
    }
    all_events: dict[str, Any] = {}
    for source, scenes in zones.items():
        events = {}
        for vid in ids:
            z = np.load(Path(c["hands"]) / f"{vid}.npz")
            ev = run_pipeline(load_video(z, vid, "ucf"), hold_of(z, c), scenes[vid], c, [])
            events[vid] = {**ev, "n_frames": int(z["n_frames"])}
        all_events[source] = events
        gap = round(c["merge_seconds"] * c["fps"])
        for variant, key in VARIANTS.items():
            out: dict[str, Any] = {}
            for name in ("official", "holdout"):
                out[name] = score_set(events, spans[name], key, c)
            hours = sum(events[v]["n_frames"] for v in normal) / c["fps"] / 3600
            n = sum(len(merged([a["frame"] for a in events[v][key]], gap)) for v in normal)
            out["normal"] = {"videos": len(normal), "hours": hours, "alerts": n,
                             "alerts_per_hour": n / hours}  # fmt: skip
            out["contacts_theft_videos"] = sum(
                len(events[v]["contacts"]) for v in spans["official"] | spans["holdout"]
            )
            out["contacts_normal_videos"] = sum(len(events[v]["contacts"]) for v in normal)
            results[f"{source}/{variant}"] = out
    (run / "events.json").write_text(json.dumps(all_events, indent=1), encoding="utf-8")
    (run / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    brief = {
        k: {s: {m: v[s][m] for m in ("events", "events_detected", "alerts", "alerts_hitting",
                                     "null_hitting_mean", "null_hitting_p", "alerts_per_hour")}
            for s in ("official", "holdout")} | {"normal_per_hour": v["normal"]["alerts_per_hour"]}
        for k, v in results.items() if "/" in k
    }  # fmt: skip
    print(json.dumps({"zone_agreement": results["zone_agreement"], **brief}, indent=1))
    print("run", run)


if __name__ == "__main__":
    main()
