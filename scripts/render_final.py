"""Session 3, E.12: skeleton video and timeline of the test videos with theft.

Rule pre-declared in configs/frozen/PREDECLARED.md: after `--final`, every test video
with an annotated theft is rendered (skeletons only, no raw video; PoseLift has none),
with the operational score of the main model (seed 0) at the main operating point,
threshold, alerts, annotated intervals and not-scorable spans; the slides use the first
two in alphabetical order. Nothing is chosen by looking at the results. `select: all` is
only for a dry run on a validation evaluation.

Run as a module, because it reuses the 2D demo drawing: `python -m scripts.render_final`.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.demo_score import _people, _timeline
from tcc.data.poselift import load_video_with_real
from tcc.eval.alerts import alerts
from tcc.eval.events import label_intervals
from tcc.pipeline.scoring import (
    MOST_NORMAL,
    NO_WINDOW,
    NOT_SCORABLE,
    SCORED,
    FrameScores,
    last_frame_reasons,
    last_frame_scores,
)
from tcc.preprocess.quality import REASONS
from tcc.utils.config import load_config
from tcc.utils.render import annotate_frame, blank_canvas
from tcc.utils.run_logger import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/render_final.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    ev = Path(config["evaluate_run"])
    results = json.loads((ev / "results.json").read_text(encoding="utf-8"))
    model_run = Path(results["per_model"][config["model_index"]]["run"])
    c1 = json.loads((model_run / "meta.json").read_text(encoding="utf-8"))["config"]
    t, dataset = c1["treatment"], c1["dataset"]
    eval_config = json.loads((ev / "meta.json").read_text(encoding="utf-8"))["config"]
    a = eval_config["alerts"]
    n = str(config["operating_point"])
    threshold = results["per_model"][config["model_index"]]["thresholds"][n]["threshold"]
    config["source"] = {
        "model_run": str(model_run),
        "threshold": threshold,
        "split": results["split"],
    }
    run_dir = start_run(f"render_final_{results['split']}", config=config, seeds={})

    z = np.load(ev / f"frames_model{config['model_index']}.npz")
    ids = sorted(str(v) for v in z["ids"])
    if config["select"] == "theft":
        ids = [v for v in ids if (z[f"{v}__labels"] == 1).any()]
    ids = ids[: config.get("limit") or len(ids)]
    split = "test" if results["split"] == "test" else "train"
    fps = float(dataset["fps"])
    size = (int(dataset["image_size"][0]), int(dataset["image_size"][1]))
    scale, T = float(config["scale"]), int(t["T"])
    summary = []
    for vid in ids:
        last, reasons = z[f"{vid}__last"], z[f"{vid}__reasons_last"].astype(np.int8)
        scored = np.isfinite(last)
        state = np.where(scored, SCORED, np.where(reasons != 0, NOT_SCORABLE, NO_WINDOW))
        fs = FrameScores(np.where(scored, last, MOST_NORMAL), state.astype(np.int8), reasons)
        found = alerts(last, scored, threshold, a["min_alert_frames"], a["merge_gap_frames"])
        theft = label_intervals(z[f"{vid}__labels"])
        _timeline(run_dir / f"{vid}_timeline.png", f"{vid} (operational, N = {n}/h)", fs, fps,
                  threshold, found, theft)  # fmt: skip

        meta, scores = z[f"{vid}__window_meta"], z[f"{vid}__window_scores"]
        ok = np.isfinite(scores)
        per_person = last_frame_scores(meta[ok], scores[ok].astype(np.float32), T)
        per_reason = last_frame_reasons(meta, z[f"{vid}__window_reason"].astype(np.int8), T)
        poses, _ = load_video_with_real(Path(dataset["root"]), split, vid, dataset)
        people = _people(poses, t)
        alert_frames = np.zeros(len(last), np.bool_)
        for al in found:
            alert_frames[al.fired : al.end] = True
        frame_size = (int(size[0] * scale), int(size[1] * scale))
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(run_dir / f"{vid}_skeleton.mp4"), fourcc, fps, frame_size)
        background = blank_canvas(size, scale)
        for q in range(len(last)):
            rows = people.get(q, [])
            tids = np.array([r[0] for r in rows], np.int64)
            kps = np.array([r[1] for r in rows], np.float32).reshape(-1, 17, 3)
            boxes = np.array([r[2] for r in rows], np.float32).reshape(-1, 4)
            interp = np.array([r[3] for r in rows], np.bool_)
            s = np.array([per_person.get((int(k), q), np.nan) for k in tids])
            why = [REASONS[per_reason[(int(k), q)]] if (int(k), q) in per_reason else ""
                   for k in tids]  # fmt: skip
            tag = " [furto anotado]" if any(x <= q < y for x, y in theft) else ""
            cap = f"{vid}  t={q / fps:.1f}s  thr={threshold:.4f}{tag}"
            writer.write(annotate_frame(background, kps, tids, boxes, s, threshold, scale,
                                        config["conf_min_draw"], bool(alert_frames[q]), cap,
                                        interpolated=interp, reasons=why))  # fmt: skip
        writer.release()
        summary.append({"video": vid, "alerts": [al._asdict() for al in found],
                        "theft_intervals": theft,
                        "not_scorable_frames": int((state == NOT_SCORABLE).sum())})  # fmt: skip
    out = {"videos": summary, "slides": ids[:2], "threshold": threshold, "operating_point": n}
    (run_dir / "renders.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"rendered": ids, "slides": ids[:2]}, indent=1))
    print(run_dir)


if __name__ == "__main__":
    main()
