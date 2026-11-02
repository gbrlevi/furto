"""Online triage: pose process -> per-track zone visits -> unpaid-exit alerts, as they happen.

Starts scripts/online_pose.py in the pose environment (same protocol as online_score.py),
feeds every frame to tcc.rules.zones.ExitMonitor and writes one JSON line per alert, with
the wall-clock delay between the exit frame being decoded and the alert. Then checks that
the offline rule on the same records gives the same alerts.
"""

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tcc.data.schema import VideoPoses
from tcc.rules.zones import ExitMonitor, ExitRule, track_visits, unpaid_exits
from tcc.utils.config import load_config
from tcc.utils.run_logger import start_run


def run_video(video: dict[str, Any], config: dict[str, Any], run: Path) -> dict[str, Any]:
    zones_cfg = load_config(Path(video["zones"]))
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    cmd = [config["pose_python"], "scripts/online_pose.py", "--config", config["pose_config"]]
    stderr = open(run / f"{video['id']}_pose_stderr.log", "w", encoding="utf-8")  # noqa: SIM115
    child = subprocess.Popen(
        [*cmd, "--video", video["path"]],
        stdout=subprocess.PIPE,
        stderr=stderr,
        env=env,
        text=True,
        encoding="utf-8",
    )
    assert child.stdout is not None
    header = json.loads(child.stdout.readline())
    fps = float(header["fps"])
    t, r = zones_cfg["tracking"], zones_cfg["rule"]
    rule = ExitRule(
        round(r["min_shelf_seconds"] * fps),
        round(r["min_checkout_seconds"] * fps),
        round(r["payment_grace_seconds"] * fps),
    )
    args = (
        zones_cfg["zones"],
        t["min_ankle_conf"],
        round(t["max_gap_seconds"] * fps),
        round(t["min_visit_seconds"] * fps),
    )
    monitor = ExitMonitor(*args, rule)
    payments = []
    if zones_cfg.get("payments"):
        with open(zones_cfg["payments"], newline="", encoding="utf-8") as f:
            payments = sorted(round(float(row["seconds"]) * fps) for row in csv.DictReader(f))
    read_time: dict[int, float] = {}
    rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    found: list[dict[str, Any]] = []
    n_frames = 0
    with open(run / f"{video['id']}_alerts.jsonl", "w", encoding="utf-8") as out:

        def emit(alerts: list[Any], now: float) -> None:
            for a in alerts:
                line = {
                    **asdict(a),
                    "seconds": a.frame / fps,
                    "delay_after_exit_seconds": now - read_time[a.frame],
                }
                found.append(line)
                out.write(json.dumps(line) + "\n")

        for raw in child.stdout:
            msg = json.loads(raw)
            if msg["type"] == "end":
                break
            q = int(msg["q"])
            n_frames = q + 1
            read_time[q] = float(msg["read_time"])
            while payments and payments[0] <= q:
                monitor.pay(payments.pop(0))
            tid = np.asarray(msg["track_id"], np.int64)
            box = np.asarray(msg["bbox"], np.float32).reshape(-1, 4)
            kp = np.asarray(msg["keypoints"], np.float32).reshape(-1, 17, 3)
            rows.append((q, tid, box, kp))
            emit(monitor.push(q, tid, box, kp), time.time())
        emit(monitor.flush(), time.time())
    child.wait()
    video_poses = VideoPoses(
        video_id=video["id"],
        camera=video["camera"],
        n_frames=n_frames,
        frame_idx=np.concatenate([np.full(len(t_), q, np.int64) for q, t_, _, _ in rows]),
        track_id=np.concatenate([t_ for _, t_, _, _ in rows]),
        bbox=np.concatenate([b for _, _, b, _ in rows]),
        keypoints=np.concatenate([k for _, _, _, k in rows]),
        frame_labels=None,
    )
    np.savez_compressed(
        run / f"{video['id']}_poses.npz",
        frame_idx=video_poses.frame_idx,
        track_id=video_poses.track_id,
        bbox=video_poses.bbox,
        keypoints=video_poses.keypoints,
        n_frames=n_frames,
    )
    all_payments = [p for p in monitor.payments]
    offline = unpaid_exits(track_visits(video_poses, *args), all_payments, rule)
    return {
        "frames": n_frames,
        "alerts": found,
        "offline_alert_frames": [a.frame for a in offline],
        "online_equals_offline": [a["frame"] for a in found] == [a.frame for a in offline],
    }


def main() -> None:
    config = load_config(Path(sys.argv[1]))
    run = start_run("triage_online", config=config, seeds={})
    results = {v["id"]: run_video(v, config, run) for v in config["videos"]}
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1))
    print("run", run)


if __name__ == "__main__":
    main()
