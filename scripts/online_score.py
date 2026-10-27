"""Online pipeline up to "score > threshold" (session 3B, task 6): the scoring process.

Main environment. For each video it starts scripts/online_pose.py in the pose environment,
reads its records from the pipe as they come, keeps a queue of the last frames of each
person (tcc.pipeline.online), applies the production treatment of the C1 run (scale guard
and quality gate), scores each window on its last frame, compares the frame score with the
threshold and applies the alert logic of session 2D. Every final frame goes to the event
log with its state ("not scorable" with the reason); alerts are logged when they fire.

Measures the end-to-end latency (frame decoded -> its own windows scored; and frame
decoded -> frame final, which waits k frames for late interpolated windows) and the
sustained fps. Then recomputes the same frames offline with the same checkpoint and
records the equivalence. Draws the timeline and the skeleton video of session 2D; the video
over real frames goes to `internal_dir` (data/, never versioned; ADR-0005).
"""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcc.data.schema import VideoPoses
from tcc.eval.alerts import alerts
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.pipeline.online import AlertLog, FinalFrame, OnlineScorer
from tcc.pipeline.resample import frames_to_keep
from tcc.pipeline.scoring import (
    MOST_NORMAL,
    NOT_SCORABLE,
    SCORED,
    STATES,
    FrameScores,
    frame_scores,
    last_frame_reasons,
    last_frame_scores,
)
from tcc.preprocess.quality import REASONS, SCORABLE
from tcc.preprocess.windows import treated_windows
from tcc.utils.config import load_config
from tcc.utils.demo_video import people_by_frame, render_videos, timeline
from tcc.utils.run_logger import start_run


def _quantiles(ms: list[float]) -> dict[str, Any]:
    if not ms:
        return {"n": 0}
    v = np.asarray(ms)
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "p50": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def _offline(
    video: VideoPoses,
    t: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    loss_norm: str,
    size: tuple[int, int],
    batch: int,
) -> tuple[Any, Any, FrameScores]:
    w = treated_windows(video, t, 1, size)
    ok = w.reason == SCORABLE
    scores = window_scores(model, w.x[ok], device, loss_norm, batch)
    per_person = last_frame_scores(w.meta[ok], scores, t["T"])
    per_reason = last_frame_reasons(w.meta, w.reason, t["T"])
    return per_person, per_reason, frame_scores(per_person, per_reason, video.n_frames)


def _run_video(
    video_cfg: dict[str, Any],
    config: dict[str, Any],
    c1: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    t, loss_norm, batch = c1["treatment"], c1["loss"]["loss_norm"], config["batch_size"]
    threshold = float(config["threshold"])
    vid = video_cfg["id"]
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    cmd = [
        config["pose_python"],
        "scripts/online_pose.py",
        "--config",
        config["pose_config"],
        "--video",
        video_cfg["path"],
    ]
    stderr = open(run_dir / f"{vid}_pose_stderr.log", "w", encoding="utf-8")  # noqa: SIM115
    child = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=stderr, env=env, text=True, encoding="utf-8"
    )
    assert child.stdout is not None
    header = json.loads(child.stdout.readline())
    size = (int(header["source_size"][0]), int(header["source_size"][1]))
    fps = float(header["fps"])
    scorer = OnlineScorer(model, device, t, size, loss_norm, batch)
    log = AlertLog(threshold, config["min_alert_frames"], config["merge_gap_frames"])
    read_time: dict[int, float] = {}
    rows_q, rows_t, rows_b, rows_k = [], [], [], []
    own_latency, final_latency, events = [], [], []
    finals: list[FinalFrame] = []
    start_wall = None
    end_message: dict[str, Any] = {}
    with open(run_dir / f"{vid}_events.jsonl", "w", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    "type": "start",
                    "video": vid,
                    "c1_run": config["c1_run"],
                    "threshold": threshold,
                    "threshold_frozen": False,
                    "threshold_origin": config["threshold_origin"],
                }
            )
            + "\n"
        )

        def emit(done: list[FinalFrame], now: float) -> None:
            for f in done:
                final_latency.append(1000 * (now - read_time[f.frame]))
                new = log.add(f)
                alert_on = bool(log.previous and log.previous[-1].end == f.frame + 1)
                line = {
                    "type": "frame",
                    "frame": f.frame,
                    "seconds": f.frame / fps,
                    "state": STATES[f.state],
                    "score": f.score if f.state == SCORED else None,
                    "reason": REASONS[f.reason] if f.state == NOT_SCORABLE else None,
                    "above": bool(f.state == SCORED and f.score > threshold),
                    "alert_active": alert_on,
                    "wall_time": now,
                }
                out.write(json.dumps(line) + "\n")
                for e in new:
                    e = {"type": "event", **e, "wall_time": now}
                    events.append(e)
                    out.write(json.dumps(e) + "\n")
            finals.extend(done)

        for raw in child.stdout:
            msg = json.loads(raw)
            if msg["type"] == "end":
                end_message = msg
                break
            q = int(msg["q"])
            read_time[q] = float(msg["read_time"])
            start_wall = start_wall or read_time[q]
            tid = np.asarray(msg["track_id"], np.int64)
            box = np.asarray(msg["bbox"], np.float32).reshape(-1, 4)
            kp = np.asarray(msg["keypoints"], np.float32).reshape(-1, 17, 3)
            rows_q.append(np.full(len(tid), q, np.int64))
            rows_t.append(tid)
            rows_b.append(box)
            rows_k.append(kp)
            before = len(scorer.per_person)
            done = scorer.push(q, tid, box, kp)
            now = time.time()
            if any(g == q for (_, g) in list(scorer.per_person)[before:]):
                own_latency.append(1000 * (now - read_time[q]))
            emit(done, now)
        emit(scorer.flush(), time.time())
    child.wait()
    stderr.close()
    n = len(finals)
    last_wall = time.time()

    video = VideoPoses(
        vid,
        video_cfg["camera"],
        n,
        np.concatenate(rows_q),
        np.concatenate(rows_t),
        np.concatenate(rows_b),
        np.concatenate(rows_k),
        None,
    )
    np.savez_compressed(
        run_dir / f"{vid}_poses.npz",
        frame_idx=video.frame_idx,
        track_id=video.track_id,
        bbox=video.bbox,
        keypoints=video.keypoints,
        n_frames=n,
    )
    per_person, per_reason, fs = _offline(video, t, model, device, loss_norm, size, batch)
    state = np.array([f.state for f in finals], np.int8)
    score = np.array([f.score for f in finals])
    reason = np.array([f.reason for f in finals], np.int8)
    common = sorted(per_person.keys() & scorer.per_person.keys())
    rel = [abs(scorer.per_person[k] - per_person[k]) / abs(per_person[k]) for k in common]
    scored = fs.state == SCORED
    offline_alerts = alerts(
        fs.score, scored, threshold, config["min_alert_frames"], config["merge_gap_frames"]
    )
    online_alerts = log.alerts()
    equivalence = {
        "window_keys_equal": per_person.keys() == scorer.per_person.keys(),
        "not_scorable_keys_and_reasons_equal": per_reason == scorer.per_reason,
        "frame_states_equal": bool((state == fs.state).all()),
        "frame_reasons_equal": bool((reason == fs.reason).all()),
        "max_relative_window_score_difference": float(max(rel)) if rel else 0.0,
        "tolerance_declared": config["equivalence_rtol"],
        "within_tolerance": bool(max(rel, default=0.0) <= config["equivalence_rtol"]),
        "alerts_equal": online_alerts == offline_alerts,
    }
    online_fs = FrameScores(np.where(state == SCORED, score, MOST_NORMAL), state, reason)

    keep = frames_to_keep(header["source_fps"], header["source_frames"], fps)[:n]
    alert_frames = np.zeros(n, np.bool_)
    for a in online_alerts:
        alert_frames[a.start : a.end] = True
    timeline(
        run_dir / f"{vid}_timeline.png",
        f"{vid}: online pipeline (demonstration)",
        online_fs,
        fps,
        threshold,
        online_alerts,
        [],
    )
    internal = Path(config["internal_dir"])
    internal.mkdir(parents=True, exist_ok=True)
    render_videos(
        {**video_cfg, "id": vid},
        people_by_frame(video, t),
        keep,
        scorer.per_person,
        scorer.per_reason,
        online_fs,
        alert_frames,
        threshold,
        [],
        fps,
        size,
        config["conf_min_draw"],
        run_dir / f"{vid}_public.mp4",
        internal / f"{vid}_internal.mp4",
    )
    with open(run_dir / f"{vid}_events.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "score", "state", "reason", "alert"])
        for k in range(n):
            writer.writerow(
                [
                    k,
                    f"{k / fps:.3f}",
                    f"{score[k]:.6g}" if state[k] == SCORED else "",
                    STATES[int(state[k])],
                    REASONS[int(reason[k])] if reason[k] else "",
                    int(alert_frames[k]),
                ]
            )

    lag = [read_time[q] - (start_wall + q / fps) for q in sorted(read_time)] if start_wall else []
    pose_seconds = end_message.get("seconds", {})
    return {
        "video": vid,
        "output_frames": n,
        "seconds_of_video": n / fps,
        "persons_per_frame": float(len(video) / max(n, 1)),
        "latency_ms_frame_to_own_score": _quantiles(own_latency),
        "latency_ms_frame_to_final": _quantiles(final_latency),
        "final_delay_frames": scorer.k,
        "sustained_fps": n / (last_wall - start_wall) if start_wall else None,
        "lag_behind_video_seconds": {
            "last": lag[-1] if lag else None,
            "max": max(lag) if lag else None,
        },
        "pose_process": {
            "header": header,
            "stage_seconds": pose_seconds,
            "stage_fps": {k: n / v for k, v in pose_seconds.items() if v},
        },
        "frames_scored": int((state == SCORED).sum()),
        "frames_not_scorable": int((state == NOT_SCORABLE).sum()),
        "frames_above_threshold": int(((state == SCORED) & (score > threshold)).sum()),
        "alerts": [
            {
                "start": a.start,
                "end": a.end,
                "fired": a.fired,
                "seconds": [a.start / fps, a.end / fps],
            }
            for a in online_alerts
        ],
        "events": events,
        "equivalence_with_offline": equivalence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/online_score.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    c1_run = Path(config["c1_run"])
    c1 = json.loads((c1_run / "meta.json").read_text(encoding="utf-8"))["config"]
    config["c1_treatment"] = c1["treatment"]
    config["pose"] = load_config(Path(config["pose_config"]))
    pose = config["pose"]
    models = {
        "autoencoder": {
            "name": "C1 ST-GCN autoencoder",
            "version": c1_run.name,
            "weights": str(c1_run / "model.pt"),
        },
        "detector": {
            "name": pose["detector"]["name"],
            "version": "see pose_env_versions",
            "weights": pose["detector"]["weights"],
        },
        "pose": {
            "name": pose["pose"]["name"],
            "version": "see pose_env_versions",
            "weights": pose["pose"]["weights"],
        },
    }
    run_dir = start_run(config["name"], config=config, seeds={}, models=models)
    ck = torch.load(c1_run / "model.pt", map_location="cpu", weights_only=True)
    m = ck["model"]
    model = STGCNAutoencoder(m["channels"], m["latent"], m["k_t"], m["dropout"])
    model.load_state_dict(ck["state"], strict=True)
    device = torch.device(config["device"])
    model.to(device).eval()
    results = {"c1_run": str(c1_run), "threshold": config["threshold"], "videos": {}}
    for video_cfg in config["videos"]:
        if not isinstance(video_cfg["id"], str):
            raise SystemExit(f"video ids must be quoted strings, got {video_cfg['id']!r}")
        r = _run_video(video_cfg, config, c1, model, device, run_dir)
        results["videos"][video_cfg["id"]] = r
        show = {
            k: r[k]
            for k in [
                "latency_ms_frame_to_own_score",
                "latency_ms_frame_to_final",
                "sustained_fps",
                "alerts",
                "equivalence_with_offline",
            ]
        }
        print(video_cfg["id"], json.dumps(show), flush=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
