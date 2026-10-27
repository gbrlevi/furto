"""Online scoring of pose records, frame by frame (session 3B, task 6).

The offline treatment is applied to a short buffer of each person, so the windows, scores
and "not scorable" states are the same as offline (`treated_windows` on the whole video):

- a window only depends on the person's observations from T + 2k frames before its end
  (gaps up to k frames are interpolated between their two neighbours, longer gaps cut);
- a gap of up to k frames is known only when the person reappears, so windows ending on
  interpolated frames arrive late, at most k frames after their last frame. Frame q is
  therefore final when frame q + k has been processed (k = `max_gap`; 8 frames = 0.53 s);
- a person not seen for more than k frames is dropped: if it comes back, a new piece starts,
  as offline.

The frame score is the maximum over people of the window scores on their last frame
(causal operation mode, ADR-0008), with the frame states of ADR-0014. Alerts use
`tcc.eval.alerts` on the final frames.
"""

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from tcc.data.schema import VideoPoses
from tcc.eval.alerts import Alert, alerts
from tcc.model.train import window_scores
from tcc.pipeline.scoring import SCORED, frame_scores
from tcc.preprocess.quality import SCORABLE
from tcc.preprocess.windows import treated_windows


class FinalFrame(NamedTuple):
    frame: int
    score: float  # MOST_NORMAL unless SCORED
    state: int  # NO_WINDOW, SCORED or NOT_SCORABLE (tcc.pipeline.scoring)
    reason: int  # quality-gate reason of NOT_SCORABLE frames (tcc.preprocess.quality)


class _Buffer:
    """Recent observations of one person."""

    def __init__(self) -> None:
        self.frames: list[int] = []
        self.boxes: list[npt.NDArray[np.float32]] = []
        self.keypoints: list[npt.NDArray[np.float32]] = []

    def add(self, q: int, box: npt.NDArray[np.float32], kp: npt.NDArray[np.float32]) -> None:
        self.frames.append(q)
        self.boxes.append(box)
        self.keypoints.append(kp)

    def trim(self, first: int) -> None:
        keep = next((i for i, q in enumerate(self.frames) if q >= first), len(self.frames))
        del self.frames[:keep], self.boxes[:keep], self.keypoints[:keep]


class OnlineScorer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        treatment: dict[str, Any],  # Any: the `treatment` section of the C1 config
        image_size: tuple[int, int],
        loss_norm: str,
        batch_size: int,
    ) -> None:
        self.model, self.device, self.t = model, device, treatment
        self.size, self.loss_norm, self.batch_size = image_size, loss_norm, batch_size
        self.T, self.k = int(treatment["T"]), int(treatment["max_gap"])
        self.horizon = self.T + 2 * self.k + 1  # frames of history a new window can need
        self.buffers: dict[int, _Buffer] = {}
        self.last_seen: dict[int, int] = {}
        self.per_person: dict[tuple[int, int], float] = {}  # (track, last frame) -> score
        self.per_reason: dict[tuple[int, int], int] = {}  # (track, last frame) -> reason
        self.next_final = 0
        self.last_q = -1

    def active_ids(self) -> list[int]:
        return sorted(self.buffers)

    def push(
        self,
        q: int,
        track_ids: npt.NDArray[np.int64],
        boxes: npt.NDArray[np.float32],
        keypoints: npt.NDArray[np.float32],
    ) -> list[FinalFrame]:
        """Add the records of frame q (frames must come in order, every frame once, also
        when nobody is there); score the windows that became complete; return the frames
        that became final (q - k, or none at the start)."""
        if q != self.last_q + 1:
            raise ValueError(f"frames must come in order: expected {self.last_q + 1}, got {q}")
        self.last_q = q
        blocks, keys = [], []
        for tid, box, kp in zip(track_ids.tolist(), boxes, keypoints, strict=True):
            buf = self.buffers.setdefault(tid, _Buffer())
            previous = self.last_seen.get(tid, -1)
            buf.add(q, box, kp)
            buf.trim(q - self.horizon)
            self.last_seen[tid] = q
            w = treated_windows(self._video(tid, buf, q), self.t, 1, self.size)
            last = w.meta[:, 2] + self.T - 1
            new = last > previous
            for r, f in zip(w.reason[new].tolist(), last[new].tolist(), strict=True):
                if r != SCORABLE:
                    self.per_reason[(tid, f)] = r
            ok = new & (w.reason == SCORABLE)
            blocks.append(w.x[ok])
            keys += [(tid, f) for f in last[ok].tolist()]
        if keys:
            x = np.concatenate(blocks)
            scores = window_scores(self.model, x, self.device, self.loss_norm, self.batch_size)
            for key, s in zip(keys, scores.tolist(), strict=True):
                if np.isfinite(s):
                    self.per_person[key] = float(s)
        for tid in [t for t, seen in self.last_seen.items() if q - seen > self.k]:
            del self.buffers[tid], self.last_seen[tid]  # gone for more than k frames
        return self._finalize(q - self.k)

    def flush(self) -> list[FinalFrame]:
        """End of the video: every remaining frame is final."""
        return self._finalize(self.last_q)

    def _video(self, tid: int, buf: _Buffer, q: int) -> VideoPoses:
        n = len(buf.frames)
        return VideoPoses(
            video_id="online", camera="online", n_frames=q + 1,
            frame_idx=np.asarray(buf.frames, np.int64), track_id=np.full(n, tid, np.int64),
            bbox=np.stack(buf.boxes).astype(np.float32),
            keypoints=np.stack(buf.keypoints).astype(np.float32), frame_labels=None,
        )  # fmt: skip

    def _finalize(self, upto: int) -> list[FinalFrame]:
        out = []
        while self.next_final <= upto:
            f = self.next_final
            people = {(t, 0): s for (t, g), s in self.per_person.items() if g == f}
            reasons = {(t, 0): r for (t, g), r in self.per_reason.items() if g == f}
            fs = frame_scores(people, reasons, 1)
            out.append(FinalFrame(f, float(fs.score[0]), int(fs.state[0]), int(fs.reason[0])))
            self.next_final += 1
        return out


class AlertLog:
    """Alerts of `tcc.eval.alerts` on the final frames seen so far, as events."""

    def __init__(self, threshold: float, min_alert_frames: int, merge_gap_frames: int) -> None:
        self.threshold, self.min, self.merge = threshold, min_alert_frames, merge_gap_frames
        self.score: list[float] = []
        self.scored: list[bool] = []
        self.previous: list[Alert] = []

    def alerts(self) -> list[Alert]:
        return alerts(
            np.asarray(self.score), np.asarray(self.scored, np.bool_), self.threshold, self.min,
            self.merge,
        )  # fmt: skip

    def add(self, frame: FinalFrame) -> list[dict[str, Any]]:
        """Append one final frame; return "alert_fired" when an alert reaches the minimum
        duration and "alert_merged" when a new run joins the previous alert."""
        if frame.frame != len(self.score):
            raise ValueError("final frames must come in order")
        self.score.append(frame.score)
        self.scored.append(frame.state == SCORED)
        current = self.alerts()
        events: list[dict[str, Any]] = []
        if len(current) > len(self.previous):
            a = current[-1]
            events.append({"event": "alert_fired", "start": a.start, "fired": a.fired,
                           "at_frame": frame.frame})  # fmt: skip
        elif current and self.previous and current[-1].end > self.previous[-1].end + 1:
            events.append({"event": "alert_merged", "start": current[-1].start,
                           "at_frame": frame.frame})  # fmt: skip
        self.previous = current
        return events
