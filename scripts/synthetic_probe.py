"""Session 4, item 5: synthetic probe over the 10 generated videos (qualitative, not evaluation).

Main: the online pipeline of session 3B with the seed 0 checkpoint and its N = 5 threshold.
Complementary: the same poses scored offline with seeds 1 to 4 and their own thresholds. The
alert and hit rules are `tcc.eval.alerts` and `tcc.eval.events.event_hits`. Nothing in
`configs/frozen/` changes. Run as a module: `python -m scripts.synthetic_probe`.
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.online_score import _run_video
from tcc.data.synthetic import (
    annotation_files,
    check_annotations_before,
    read_annotations,
    read_manifest,
    reference_intervals,
    reference_labels,
    validate_annotations,
    validate_manifest,
)
from tcc.eval.alerts import Alert, alerts
from tcc.eval.events import event_hits
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.pipeline.scoring import (
    MOST_NORMAL,
    NO_WINDOW,
    NOT_SCORABLE,
    SCORED,
    STATES,
    FrameScores,
    frame_scores,
    last_frame_reasons,
    last_frame_scores,
    load_extracted,
)
from tcc.preprocess.quality import REASONS, SCORABLE
from tcc.preprocess.windows import treated_windows
from tcc.utils.config import load_config
from tcc.utils.demo_video import timeline
from tcc.utils.frozen import sha256
from tcc.utils.run_logger import start_run

COLUMNS = [
    "id", "action_appeared", "artifacts", "frames", "frames_with_person", "frames_scored",
    "scored_pct_of_frames_with_person", "frames_not_scorable", "top_not_scorable_reason",
    "max_score", "threshold", "max_score_over_threshold", "alerts_seed0", "alert_hit_seed0",
    "hit_interval_source", "seeds_1_4_with_alert",
]  # fmt: skip


def _model(run: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    c1 = json.loads((run / "meta.json").read_text(encoding="utf-8"))["config"]
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    m = ck["model"]
    net = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
    net.load_state_dict(ck["state"], strict=True)
    net.to(device).eval()
    return net, c1


def _group_data(
    config: dict[str, Any], ids: list[str], root: Path, started_at: datetime
) -> dict[str, Any]:
    """Manifest and annotations if the group has filled them; what is missing is recorded."""
    out: dict[str, Any] = {"deviations": [], "annotations": {}, "reference": None}
    manifest = root / "manifest.csv"
    if manifest.exists():
        if config["frozen_at"] is None:
            raise SystemExit("manifest.csv exists but run.yaml has no freeze date (frozen_at)")
        frozen_at = datetime.fromisoformat(config["frozen_at"])
        validate_manifest(read_manifest(manifest), ids, frozen_at)
    else:
        out["deviations"].append("manifest.csv absent when the pipeline ran")
    if len(list(root.glob("annotations_*.csv"))) == 2:
        paths = annotation_files(root)
        rows = [read_annotations(p) for p in paths]
        for r in rows:
            validate_annotations(r, ids)
        out["annotation_mtimes"] = check_annotations_before(paths, started_at)
        out["reference"] = reference_intervals(rows[0], rows[1])
        out["annotations"] = {
            r["id"]: {
                "action": "/".join(x[r["id"]]["action_appeared"] for x in _by_id(rows)),
                "artifacts": "/".join(x[r["id"]]["artifacts"] for x in _by_id(rows)),
            }
            for r in rows[0]
        }
    else:
        out["deviations"].append(
            "annotations absent when the pipeline ran: hit column uses the prompted interval"
        )
    return out


def _by_id(rows: list[list[dict[str, str]]]) -> list[dict[str, dict[str, str]]]:
    return [{r["id"]: r for r in x} for x in rows]


def _offline(
    video: Any, c1: dict[str, Any], net: torch.nn.Module, device: torch.device,
    size: tuple[int, int], batch: int,
) -> FrameScores:  # fmt: skip
    t = c1["treatment"]
    w = treated_windows(video, t, 1, size)
    ok = w.reason == SCORABLE
    scores = window_scores(net, w.x[ok], device, c1["loss"]["loss_norm"], batch)
    return frame_scores(
        last_frame_scores(w.meta[ok], scores, t["T"]),
        last_frame_reasons(w.meta, w.reason, t["T"]),
        video.n_frames,
    )


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(str(r[c]) for c in COLUMNS) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_probe/run.yaml"))
    config = load_config(parser.parse_args().config)
    frozen = load_config(Path(config["frozen_config"]))
    prompts = load_config(Path(config["prompts"]))
    thresholds = json.loads(
        (Path(config["thresholds_run"]) / "results.json").read_text(encoding="utf-8")
    )["per_model"]
    op = str(config["operating_point"])
    seed_thresholds = [m["thresholds"][op]["threshold"] for m in thresholds]
    if round(seed_thresholds[0], 6) != config["expected_seed0_threshold"]:
        raise SystemExit(f"seed 0 threshold {seed_thresholds[0]} differs from the protocol")
    for m in frozen["models"]:
        if sha256(Path(m["run"]) / "model.pt") != m["sha256"]:
            raise SystemExit(f"{m['run']}: checkpoint differs from the frozen configuration")

    root, ids = Path(config["videos_dir"]), list(prompts["order"])
    c1_runs = [Path(m["run"]) for m in frozen["models"]]
    pose = load_config(Path(config["pose_config"]))
    online_config = {
        **config,
        "c1_run": str(c1_runs[0]),
        "threshold": seed_thresholds[0],
        "threshold_origin": f"N = {op}, seed 0, {config['thresholds_run']}; frozen rule",
        "threshold_frozen": True,
        "min_alert_frames": frozen["alerts"]["min_alert_frames"],
        "merge_gap_frames": frozen["alerts"]["merge_gap_frames"],
        "device": frozen["device"],
        "batch_size": frozen["batch_size"],
        "pose": pose,
    }
    record = {**online_config, "frozen": frozen, "prompts_sha256": sha256(Path(config["prompts"]))}
    models = {
        f"autoencoder_seed{i}": {"name": "C1 ST-GCN autoencoder", "version": r.name,
                                 "weights": str(r / "model.pt")}
        for i, r in enumerate(c1_runs)
    }  # fmt: skip
    models["detector"] = {"name": pose["detector"]["name"], "version": "see pose_env_versions",
                          "weights": pose["detector"]["weights"]}  # fmt: skip
    models["pose"] = {"name": pose["pose"]["name"], "version": "see pose_env_versions",
                      "weights": pose["pose"]["weights"]}  # fmt: skip
    run_dir = start_run(config["name"], config=record, seeds={}, models=models)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    group = _group_data(config, ids, root, datetime.fromisoformat(meta["created_at"]))

    device = torch.device(frozen["device"])
    nets = [_model(r, device) for r in c1_runs]
    a, batch = frozen["alerts"], frozen["batch_size"]
    rows, details = [], {}
    fps = 15.0
    for vid in ids:
        path = root / f"{vid}.mp4"
        pair = vid.split("_")[0]
        video_cfg = {"id": vid, "camera": pair, "path": str(path),
                     "internal_scale": config["internal_scale"],
                     "public_scale": config["public_scale"]}  # fmt: skip
        r = _run_video(video_cfg, online_config, nets[0][1], nets[0][0], device, run_dir)
        n, fps = r["output_frames"], float(r["pose_process"]["header"]["fps"])
        size_list = r["pose_process"]["header"]["source_size"]
        size = (int(size_list[0]), int(size_list[1]))
        video = load_extracted(run_dir / f"{vid}_poses.npz", vid, pair)
        main_fs = _offline(video, nets[0][1], nets[0][0], device, size, batch)

        if group["reference"] is not None:
            intervals, source = group["reference"][vid], "annotated union"
        elif vid.endswith("_concealment"):
            intervals, source = [tuple(config["prompted_interval_s"])], "prompt, not annotated"
        else:
            intervals, source = [], "prompt, not annotated"
        labels = reference_labels(intervals, fps, n)
        fired = [x["fired"] for x in r["alerts"]]
        hit = event_hits([labels], [fired], frozen["events"]["after_frames"], n / fps / 3600)

        online = _events(run_dir / f"{vid}_events.csv", n)
        scored = online.state == SCORED
        theft = [(int(s * fps), min(int(np.ceil(e * fps)), n)) for s, e in intervals]
        found = [Alert(x["start"], x["end"], x["fired"]) for x in r["alerts"]]
        timeline(run_dir / f"{vid}_timeline.png",
                 f"{vid}: AI-generated video, qualitative illustration, not evaluation",
                 online, fps, seed_thresholds[0], found, theft, f"interval: {source}")  # fmt: skip

        others = {}
        for k in range(1, len(nets)):
            fs = _offline(video, nets[k][1], nets[k][0], device, size, batch)
            found_k = alerts(fs.score, fs.state == SCORED, seed_thresholds[k],
                             a["min_alert_frames"], a["merge_gap_frames"])  # fmt: skip
            top = fs.score[fs.state == SCORED]
            others[str(k)] = {
                "threshold": seed_thresholds[k],
                "alerts": [al._asdict() for al in found_k],
                "max_score": float(top.max()) if len(top) else None,
            }
        with_person = len(np.unique(video.frame_idx))
        reasons = Counter(online.reason[online.state == NOT_SCORABLE].tolist())
        top_score = float(online.score[scored].max()) if scored.any() else None
        note = group["annotations"].get(
            vid, {"action": "not annotated", "artifacts": "not annotated"}
        )
        ratio = top_score / seed_thresholds[0] if top_score is not None else None
        scored_pct = 100 * float(scored.sum()) / max(with_person, 1)
        rows.append({
            "id": vid,
            "action_appeared": note["action"],
            "artifacts": note["artifacts"],
            "frames": n,
            "frames_with_person": with_person,
            "frames_scored": int(scored.sum()),
            "scored_pct_of_frames_with_person": round(scored_pct, 1),
            "frames_not_scorable": int((online.state == NOT_SCORABLE).sum()),
            "top_not_scorable_reason": REASONS[reasons.most_common(1)[0][0]] if reasons else "",
            "max_score": round(top_score, 4) if top_score is not None else "",
            "threshold": round(seed_thresholds[0], 6),
            "max_score_over_threshold": round(ratio, 3) if ratio is not None else "",
            "alerts_seed0": len(fired),
            "alert_hit_seed0": hit["alerts_hitting"],
            "hit_interval_source": source,
            "seeds_1_4_with_alert": sum(bool(o["alerts"]) for o in others.values()),
        })  # fmt: skip
        same_states = bool((main_fs.state == online.state).all())
        details[vid] = {"online": r, "hit": hit, "intervals": intervals, "seeds_1_4": others,
                        "main_offline_states_equal": same_states}  # fmt: skip
        if pair != config["presentation_example"]:
            (run_dir / f"{vid}_public.mp4").unlink()

    with open(run_dir / "table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "table.md").write_text(_markdown(rows), encoding="utf-8")
    results = {"deviations": group["deviations"], "group": group, "table": rows, "videos": details}
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(_markdown(rows))
    print(json.dumps(group["deviations"], indent=1))
    print(run_dir)


def _events(path: Path, n: int) -> FrameScores:
    """Online frame scores, states and reasons from the event CSV of the run."""
    codes = {name: k for k, name in STATES.items()}
    names = {name: k for k, name in REASONS.items()}
    score, state, reason = np.full(n, MOST_NORMAL), np.zeros(n, np.int8), np.zeros(n, np.int8)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = int(row["frame"])
            state[k] = codes.get(row["state"], NO_WINDOW)
            if row["score"]:
                score[k] = float(row["score"])
            if row["reason"]:
                reason[k] = names[row["reason"]]
    return FrameScores(score, state, reason)


if __name__ == "__main__":
    main()
