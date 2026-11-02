"""Post-hoc checks of the main test evaluation: no-model baselines, AUC on frames with a
person, AUC inside each video against a rolled-label null, and the alert threshold sweep.

Opens the test (it reads the test poses to count people), so it goes through the gate of
ADR-0009 and is logged in runs/test_evaluations.csv. It reads the frame scores stored by the
opening of 27/10/2026 and computes no new model score.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tcc.data.poselift import load_video_with_real
from tcc.eval.alerts import alerts
from tcc.eval.events import event_hits
from tcc.eval.posthoc import (
    auc,
    events_with_room,
    people_per_frame,
    within_video_auc,
    within_video_null,
)
from tcc.utils.config import load_config
from tcc.utils.frozen import check_frozen, log_test_opening
from tcc.utils.run_logger import start_run

TEST_LOG = Path("runs") / "test_evaluations.csv"


def null_summary(observed: float, null: np.ndarray) -> dict[str, float]:
    return {
        "auc": observed,
        "null_mean": float(null.mean()),
        "null_low": float(np.quantile(null, 0.025)),
        "null_high": float(np.quantile(null, 0.975)),
        "p": float((1 + (null >= observed).sum()) / (1 + len(null))),
    }


def main() -> None:
    path = Path(sys.argv[1])
    frozen = check_frozen(path)
    config = load_config(path)
    run = start_run("posthoc_test", config={**config, "frozen": frozen}, seeds=config["null"])
    log_test_opening(TEST_LOG, run, frozen)
    source = Path(config["source_run"])
    meta = json.loads((Path(config["dataset_from"]) / "meta.json").read_text(encoding="utf-8"))
    dataset = meta["config"]["dataset"]
    seeds = [np.load(p) for p in sorted(source.glob("frames_model*.npz"))]
    ids = [str(v) for v in seeds[0]["ids"]]
    labels = [seeds[0][f"{v}__labels"].astype(np.int64) for v in ids]
    people, real_people = [], []
    for vid, y in zip(ids, labels, strict=True):
        video, real = load_video_with_real(Path(dataset["root"]), "test", vid, dataset)
        people.append(people_per_frame(video.frame_idx, len(y)))
        real_people.append(people_per_frame(video.frame_idx[real], len(y)))
    y = np.concatenate(labels)
    n = np.concatenate(people).astype(float)
    with_person = n > 0
    out: dict[str, Any] = {
        "videos": len(ids),
        "frames": len(y),
        "theft_frames": int(y.sum()),
        "frames_with_person": {
            "normal": int(with_person[y == 0].sum()),
            "theft": int(with_person[y == 1].sum()),
        },
        "baselines": {
            "people_count": auc(y, n),
            "people_count_real_detections": auc(y, np.concatenate(real_people).astype(float)),
            "person_present": auc(y, with_person.astype(float)),
            "people_count_frames_with_person": auc(y[with_person], n[with_person]),
        },
    }  # fmt: skip
    draws, seed = config["null"]["draws"], config["null"]["seed"]
    person_masks = [p > 0 for p in people]
    per_seed = []
    for z in seeds:
        offline = [z[f"{v}__offline"] for v in ids]
        last = [z[f"{v}__last"] for v in ids]
        scored = [np.isfinite(s) for s in last]
        last_all = np.concatenate(last)
        ok = np.isfinite(last_all)
        inside_off = within_video_auc(labels, offline, person_masks)
        inside_last = within_video_auc(labels, last, scored)
        per_seed.append(
            {
                "offline_all_frames": auc(y, np.concatenate(offline)),
                "offline_frames_with_person": auc(
                    y[with_person], np.concatenate(offline)[with_person]
                ),
                "last_frame_pooled": auc(y[ok], last_all[ok]),
                "last_frame_pooled_frames": int(ok.sum()),
                "within_video_offline": {
                    **null_summary(
                        inside_off[0],
                        within_video_null(labels, offline, person_masks, draws, seed),
                    ),
                    "videos": inside_off[1],
                },
                "within_video_last_frame": {
                    **null_summary(
                        inside_last[0], within_video_null(labels, last, scored, draws, seed)
                    ),
                    "videos": inside_last[1],
                },
            }
        )
    out["per_seed"] = per_seed
    for key in ("offline_all_frames", "offline_frames_with_person", "last_frame_pooled"):
        values = [s[key] for s in per_seed]
        out[f"{key}_mean_sd"] = [float(np.mean(values)), float(np.std(values, ddof=1))]
    for key in ("within_video_offline", "within_video_last_frame"):
        values = [s[key]["auc"] for s in per_seed]
        out[f"{key}_mean_sd"] = [float(np.mean(values)), float(np.std(values, ddof=1))]
    a, after = config["alerts"], config["events"]["after_frames"]
    hours = len(y) / config["fps"] / 3600
    last0 = [seeds[0][f"{v}__last"] for v in ids]
    out["events_with_room"] = sum(
        events_with_room(lab, np.isfinite(s), a["min_alert_frames"], after)
        for lab, s in zip(labels, last0, strict=True)
    )
    sweep = {}
    for thr in config["sweep_thresholds"]:
        found = [
            alerts(s, np.isfinite(s), thr, a["min_alert_frames"], a["merge_gap_frames"])
            for s in last0
        ]
        sweep[str(thr)] = event_hits(labels, [[x.fired for x in f] for f in found], after, hours)
    out["sweep_seed0"] = sweep
    out["hours"] = hours
    (run / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in out.items() if k not in ("per_seed", "sweep_seed0")}, indent=1)
    )
    for thr, e in sweep.items():
        print(
            thr,
            e["events_detected"],
            "of",
            e["events"],
            "alerts",
            e["alerts"],
            "false/h",
            round(e["false_alerts_per_hour"], 1),
        )
    print("run", run)


if __name__ == "__main__":
    main()
