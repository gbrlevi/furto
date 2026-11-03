"""Stages 1 and 2 of the item pipeline on the MERL Shopping Dataset.

Usage: merl_stages.py CONFIG SPLIT, SPLIT in dev (subjects 1-26) or test (27-41, once, with
the config frozen). Stage 1: contacts of a hand with the shelf zone against the union of the
reach / retract / hand-in-shelf labels, matched as events. Stage 2: product in hand, frames of
"inspect product" (holding) against "inspect shelf" (not touching), with the zero-shot CLIP
probability and with a logistic probe on the CLIP embedding of one hand (trained on subjects
1-20 for dev, 1-26 for test). Take: after a true contact, does the hold rise when an
"inspect product" follows within `take.gap` frames?
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from tcc.data.merl import load_labels, merge, split_of, to_fps
from tcc.data.schema import VideoPoses
from tcc.eval.intervals import match
from tcc.eval.posthoc import auc
from tcc.rules.shopping import contacts, hand_conf, hand_points, hands_in_zone
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def load(path: Path, labels: Path, fps: float) -> dict[str, Any]:
    z = np.load(path)
    vid = path.stem
    video = VideoPoses(
        video_id=vid, camera="merl", n_frames=int(z["n_frames"]),
        frame_idx=z["frame_idx"].astype(np.int64), track_id=z["track_id"].astype(np.int64),
        bbox=z["bbox"], keypoints=z["keypoints"], frame_labels=None,
    )  # fmt: skip
    spans = {k: to_fps(v, fps) for k, v in load_labels(labels / f"{vid}_label.mat").items()}
    return {"id": vid, "subject": int(vid.split("_")[0]), "video": video, "emb": z["emb"],
            "hold": z["hold"].astype(np.float64), "spans": spans}  # fmt: skip


def frame_mask(spans: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in spans.tolist():
        m[max(a, 0) : b + 1] = True
    return m


def frame_score(v: dict[str, Any], per_row: np.ndarray) -> np.ndarray:
    """Max over rows and hands of a frame; NaN where nobody is scored."""
    out = np.full(v["video"].n_frames, -np.inf)
    np.maximum.at(out, v["video"].frame_idx, np.nan_to_num(per_row, nan=-np.inf).max(1))
    out[np.isinf(out)] = np.nan
    return out


def stage1(videos: list[dict[str, Any]], c: dict[str, Any]) -> dict[str, Any]:
    ct, tr = c["contact"], c["truth"]
    totals = {iou: np.zeros(3, np.int64) for iou in c["match_iou"]}
    frames = np.zeros(3, np.int64)  # tp, predicted, true
    for v in videos:
        vp = v["video"]
        inside = hands_in_zone(
            hand_points(vp.keypoints, ct["push"]), hand_conf(vp.keypoints), c["shelf"],
            ct["min_conf"],
        )  # fmt: skip
        found = contacts(vp, inside, ct["max_gap"], ct["min_frames"])
        pred = merge(np.asarray([(x.start, x.end) for x in found]).reshape(-1, 2), ct["max_gap"])
        truth = merge(np.concatenate([v["spans"][a] for a in tr["actions"]]), tr["merge_gap"])
        v["contacts_true"] = truth
        for iou in c["match_iou"]:
            totals[iou] += match(pred, truth, iou)
        p = frame_mask(np.asarray(pred).reshape(-1, 2), vp.n_frames)
        t = frame_mask(np.asarray(truth).reshape(-1, 2), vp.n_frames)
        frames += [int((p & t).sum()), int(p.sum()), int(t.sum())]
    out: dict[str, Any] = {}
    for iou, (tp, fp, fn) in totals.items():
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[f"iou_{iou}"] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }  # fmt: skip
    out["frames"] = {"precision": frames[0] / max(frames[1], 1), "recall": frames[0] / frames[2]}
    return out


def boot_auc(
    videos: list[dict[str, Any]], scores: list[np.ndarray], labels: list[np.ndarray], c: dict
) -> dict[str, float]:
    keep = [np.isfinite(s) & (y >= 0) for s, y in zip(scores, labels, strict=True)]
    s = [x[k] for x, k in zip(scores, keep, strict=True)]
    y = [x[k] for x, k in zip(labels, keep, strict=True)]
    value = auc(np.concatenate(y), np.concatenate(s))
    rng = np.random.default_rng(c["seed"])
    draws = []
    for _ in range(c["null_draws"]):
        pick = rng.integers(0, len(videos), len(videos))
        draws.append(
            auc(np.concatenate([y[i] for i in pick]), np.concatenate([s[i] for i in pick]))
        )
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"auc": value, "ci_low": float(lo), "ci_high": float(hi),
            "frames_pos": int(sum((x == 1).sum() for x in y)),
            "frames_neg": int(sum((x == 0).sum() for x in y))}  # fmt: skip


def hold_labels(v: dict[str, Any]) -> np.ndarray:
    """Per frame: 1 inspect product, 0 inspect shelf, -1 otherwise (or both)."""
    n = v["video"].n_frames
    pos = frame_mask(v["spans"]["inspect_product"], n)
    neg = frame_mask(v["spans"]["inspect_shelf"], n)
    y = np.full(n, -1)
    y[pos & ~neg], y[neg & ~pos] = 1, 0
    return y


def fit_probe(videos: list[dict[str, Any]], c: dict[str, Any]) -> LogisticRegression:
    x, y = [], []
    for v in videos:
        lab = hold_labels(v)[v["video"].frame_idx]
        for hand in (0, 1):
            ok = (lab >= 0) & np.isfinite(v["hold"][:, hand])
            x.append(v["emb"][ok, hand].astype(np.float32))
            y.append(lab[ok])
    probe = LogisticRegression(C=c["probe"]["C"], max_iter=c["probe"]["max_iter"])
    return probe.fit(np.concatenate(x), np.concatenate(y))


def probe_hold(v: dict[str, Any], probe: LogisticRegression) -> np.ndarray:
    n, _, d = v["emb"].shape
    p = probe.predict_proba(v["emb"].reshape(-1, d).astype(np.float32))[:, 1].reshape(n, 2)
    return np.where(np.isfinite(v["hold"]), p, np.nan)


def youden(scores: list[np.ndarray], labels: list[np.ndarray]) -> float:
    """Threshold with the largest TPR - FPR on the frames of known class (`>=` is positive)."""
    s = np.concatenate([x[np.isfinite(x) & (y >= 0)] for x, y in zip(scores, labels, strict=True)])
    y = np.concatenate([y[np.isfinite(x) & (y >= 0)] for x, y in zip(scores, labels, strict=True)])
    fpr, tpr, thresholds = roc_curve(y, s)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def take_auc(videos: list[dict[str, Any]], frames: list[np.ndarray], c: dict) -> dict[str, Any]:
    """After each true contact: hold median in the window after minus before; label 1 when an
    "inspect product" starts within `gap` frames after the contact ends."""
    w, gap = c["take"]["window"], c["take"]["gap"]
    y, s = [], []
    for v, f in zip(videos, frames, strict=True):
        starts = v["spans"]["inspect_product"][:, 0]
        for a, b in v["contacts_true"]:
            before, after = f[max(a - w, 0) : a], f[b + 1 : b + 1 + w]
            if not (np.isfinite(before).any() and np.isfinite(after).any()):
                continue
            y.append(int(((starts > b) & (starts <= b + gap)).any()))
            s.append(float(np.nanmedian(after) - np.nanmedian(before)))
    ya, sa = np.asarray(y), np.asarray(s)
    return {"auc": auc(ya, sa), "contacts": len(ya), "took": int(ya.sum())}


def main() -> None:
    c = load_config(Path(sys.argv[1]))
    which = sys.argv[2]
    run = start_run(f"merl_stages_{which}", config={**c, "split": which}, seeds={"boot": c["seed"]})
    all_videos = [
        load(p, Path(c["labels"]), c["fps"]) for p in sorted(Path(c["hands"]).glob("*.npz"))
    ]
    split = {v["id"]: split_of(v["subject"]) for v in all_videos}
    if which == "dev":
        fit = [v for v in all_videos if split[v["id"]] == "train"]
        evaluate = [v for v in all_videos if split[v["id"]] == "val"]
        stage1_videos = fit + evaluate
    else:
        fit = [v for v in all_videos if split[v["id"]] != "test"]
        evaluate = stage1_videos = [v for v in all_videos if split[v["id"]] == "test"]
    results: dict[str, Any] = {
        "videos": {"stage1": len(stage1_videos), "probe_fit": len(fit), "evaluated": len(evaluate)},
        "stage1_contact": stage1(stage1_videos, c),
    }
    probe = fit_probe(fit, c)
    np.savez(run / "probe.npz", coef=probe.coef_, intercept=probe.intercept_)
    labels = [hold_labels(v) for v in evaluate]
    zero = [frame_score(v, v["hold"]) for v in evaluate]
    probed = [frame_score(v, probe_hold(v, probe)) for v in evaluate]
    fit_labels = [hold_labels(v) for v in fit]
    results["hold_threshold_on_fit"] = {
        "zero_shot": youden([frame_score(v, v["hold"]) for v in fit], fit_labels),
        "probe": youden([frame_score(v, probe_hold(v, probe)) for v in fit], fit_labels),
    }
    results["stage2_hold"] = {
        "zero_shot": boot_auc(evaluate, zero, labels, c),
        "probe": boot_auc(evaluate, probed, labels, c),
    }
    results["take"] = {
        "zero_shot": take_auc(evaluate, zero, c),
        "probe": take_auc(evaluate, probed, c),
    }
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1))
    print("run", run)


if __name__ == "__main__":
    main()
