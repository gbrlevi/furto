"""Timeline and skeleton videos of a scored video (sessions 2C, 2D and 3B).

Drawing only: shared by the offline demonstration (scripts/demo_score.py) and the online
pipeline (scripts/online_score.py). The video over real frames is internal (ADR-0005).
"""

from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tcc.eval.alerts import Alert
from tcc.pipeline.scoring import NOT_SCORABLE, SCORED, STATES, FrameScores
from tcc.preprocess.quality import CRITERIA, REASONS, SCORABLE
from tcc.preprocess.trajectories import split_trajectories, tracks
from tcc.utils.render import annotate_frame, blank_canvas

SCORE_COLOR, INK, GRID = "#2a78d6", "#3d3d3a", "#e4e3df"
ALERT_COLOR, THEFT_COLOR = "#e34948", "#eb6834"
# Not-scorable shading, one fixed categorical slot per criterion of CRITERIA
REASON_COLORS = dict(
    zip(CRITERIA, ["#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300"], strict=True)
)
SYMLOG_LINEAR = 0.1  # linear part of the score axis: [0, 0.1 * threshold] (drawing only)


def spans(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def timeline(
    path: Path, title: str, fs: FrameScores, fps: float, threshold: float,
    final: list[Alert], theft: list[tuple[int, int]],
) -> None:  # fmt: skip
    n = len(fs.score)
    t = np.arange(n) / fps
    fig, ax = plt.subplots(figsize=(10, 3.6))
    handles = [
        Line2D([], [], color=SCORE_COLOR, linewidth=1.5, label="frame score"),
        Line2D([], [], color=INK, linestyle="--", label=f"threshold {threshold:.4f}"),
        Patch(color=ALERT_COLOR, alpha=0.18, label="alert (>= 15 frames, merged < 2 s)"),
    ]
    for r, name in REASONS.items():
        mask = (fs.state == NOT_SCORABLE) & (fs.reason == r)
        if r == SCORABLE or not mask.any():
            continue
        for a, b in spans(mask):
            ax.axvspan(a / fps, b / fps, color=REASON_COLORS[name], alpha=0.25, linewidth=0)
        handles.append(Patch(color=REASON_COLORS[name], alpha=0.25, label=f"not scorable: {name}"))
    for al in final:
        ax.axvspan(al.start / fps, al.end / fps, color=ALERT_COLOR, alpha=0.18, linewidth=0)
    for a, b in theft:
        ax.axvspan(a / fps, b / fps, ymin=0, ymax=0.05, color=THEFT_COLOR, linewidth=0)
    if theft:
        handles.append(Patch(color=THEFT_COLOR, label="annotated theft"))
    ax.plot(t, np.where(fs.state == SCORED, fs.score, np.nan), color=SCORE_COLOR, linewidth=1.5)
    ax.axhline(threshold, color=INK, linewidth=1, linestyle="--")
    ax.set_yscale("symlog", linthresh=SYMLOG_LINEAR * threshold)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, n / fps)
    ax.set_xlabel("seconds (15 fps)", color=INK)
    ax.set_ylabel("frame score (symlog)", color=INK)
    ax.set_title(title, color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.0, 1.0))  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def people_by_frame(poses: Any, t: dict[str, Any]) -> dict[int, list[tuple[Any, ...]]]:
    """Per output frame: (track, keypoints, box, interpolated) of every detection, plus the
    poses the treatment filled inside trajectory gaps (drawn as interpolated)."""
    out: dict[int, list[tuple[Any, ...]]] = {}
    for i in range(len(poses)):
        q = int(poses.frame_idx[i])
        out.setdefault(q, []).append(
            (int(poses.track_id[i]), poses.keypoints[i], poses.bbox[i], False)
        )
    for tr in split_trajectories(tracks(poses), t["max_gap"], t["T"]):
        assert tr.real is not None and tr.boxes is not None
        for k in np.flatnonzero(~tr.real):
            q = int(tr.frames[k])
            out.setdefault(q, []).append((tr.track_id, tr.keypoints[k], tr.boxes[k], True))
    return out


def render_videos(
    video_cfg: dict[str, Any],
    people: dict[int, list[tuple[Any, ...]]],
    keep: npt.NDArray[np.int64],
    per_person: dict[tuple[int, int], float],
    per_reason: dict[tuple[int, int], int],
    fs: FrameScores,
    alert_frames: npt.NDArray[np.bool_],
    threshold: float,
    theft: list[tuple[int, int]],
    fps: float,
    size: tuple[int, int],
    conf_min: float,
    out_public: Path,
    out_internal: Path,
) -> None:
    ps, is_ = video_cfg["public_scale"], video_cfg["internal_scale"]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    public = cv2.VideoWriter(str(out_public), fourcc, fps, (int(size[0] * ps), int(size[1] * ps)))
    internal = cv2.VideoWriter(
        str(out_internal), fourcc, fps, (int(size[0] * is_), int(size[1] * is_))
    )
    background = blank_canvas(size, ps)
    cap = cv2.VideoCapture(video_cfg["path"])
    keep_pos = {int(i): k for k, i in enumerate(keep)}
    for i in range(int(keep[-1]) + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot decode frame {i} of {video_cfg['path']}")
        if i not in keep_pos:
            continue
        q = keep_pos[i]
        rows = people.get(q, [])
        tids = np.array([r[0] for r in rows], np.int64)
        kps = np.array([r[1] for r in rows], np.float32).reshape(-1, 17, 3)
        boxes = np.array([r[2] for r in rows], np.float32).reshape(-1, 4)
        interp = np.array([r[3] for r in rows], np.bool_)
        scores = np.array([per_person.get((int(k), q), np.nan) for k in tids])
        reasons = [REASONS[per_reason[(int(k), q)]] if (int(k), q) in per_reason else ""
                   for k in tids]  # fmt: skip
        tag = " [furto anotado]" if any(a <= q < b for a, b in theft) else ""
        state = int(fs.state[q])
        score = f"{fs.score[q]:.4f}" if state == SCORED else STATES[state]
        caption = f"{video_cfg['id']}  t={q / fps:.1f}s  score={score}  thr={threshold:.4f}{tag}"
        alert = bool(alert_frames[q])
        for writer, canvas, scale in [
            (public, background, ps),
            (internal, np.asarray(cv2.resize(frame, (int(size[0] * is_), int(size[1] * is_))),
                                  np.uint8), is_),
        ]:  # fmt: skip
            writer.write(
                annotate_frame(
                    canvas,
                    kps,
                    tids,
                    boxes,
                    scores,
                    threshold,
                    scale,
                    conf_min,
                    alert,
                    caption,
                    interpolated=interp,
                    reasons=reasons,
                )  # fmt: skip
            )
    public.release()
    internal.release()
