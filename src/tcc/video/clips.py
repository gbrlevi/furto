"""UCF-Crime temporal annotations and the clip layout of a video."""

import numpy as np
import numpy.typing as npt


def parse_annotations(text: str, category: str) -> dict[str, list[tuple[int, int]]]:
    """Video id -> [start, end) theft intervals, from the official annotation file.

    A row is `name.mp4 category start1 end1 start2 end2`, with -1 for no second interval.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for row in text.splitlines():
        parts = row.split()
        if len(parts) == 6 and parts[1] == category:
            values = [int(v) for v in parts[2:]]
            spans = [(a, b) for a, b in zip(values[::2], values[1::2], strict=True) if a >= 0]
            out[parts[0].removesuffix(".mp4")] = spans
    return out


def frame_labels(n_frames: int, spans: list[tuple[int, int]]) -> npt.NDArray[np.int64]:
    y = np.zeros(n_frames, np.int64)
    for a, b in spans:
        y[a:b] = 1
    return y


def clip_starts(n_frames: int, span: int, hop: int) -> npt.NDArray[np.int64]:
    """First frame of each clip of `span` frames, every `hop` frames; only complete clips."""
    return np.arange(0, max(n_frames - span + 1, 0), hop, dtype=np.int64)


def clip_frame_scores(
    starts: npt.NDArray[np.int64], scores: npt.NDArray[np.float64], span: int, n_frames: int
) -> npt.NDArray[np.float64]:
    """Causal frame scores: each clip's score holds from its last frame until the next clip's."""
    out = np.full(n_frames, np.nan)
    last = starts + span - 1
    for i, frame in enumerate(last):
        stop = last[i + 1] if i + 1 < len(last) else n_frames
        out[frame:stop] = scores[i]
    return out


def people_crop(
    boxes: npt.NDArray[np.float32],
    first: int,
    last: int,
    width: int,
    height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    """Square around all the people seen in frames [first, last]; the whole frame if nobody.

    `boxes` rows are [frame, x1, y1, x2, y2].
    """
    inside = boxes[(boxes[:, 0] >= first) & (boxes[:, 0] <= last)]
    if not len(inside):
        return 0, 0, width, height
    x1, y1 = inside[:, 1].min(), inside[:, 2].min()
    x2, y2 = inside[:, 3].max(), inside[:, 4].max()
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = min(side, width, height)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(np.clip(cx - side / 2, 0, width - side))
    top = int(np.clip(cy - side / 2, 0, height - side))
    return left, top, left + int(side), top + int(side)


def remove_scene(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Each clip minus the running mean of the earlier clips of the video (causal)."""
    if not len(x):
        return x
    earlier = np.cumsum(x, axis=0)[:-1] / np.arange(1, len(x))[:, None]
    return np.concatenate([np.zeros_like(x[:1]), x[1:] - earlier]).astype(np.float32)


def causal_standardise(
    scores: npt.NDArray[np.float64], min_history: int
) -> npt.NDArray[np.float64]:
    """Each score against the mean and deviation of the earlier ones; NaN until enough history."""
    out = np.full(len(scores), np.nan)
    for i in range(min_history, len(scores)):
        earlier = scores[:i]
        out[i] = (scores[i] - earlier.mean()) / max(earlier.std(), 1e-6)
    return out


def remove_position_trend(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Scores minus their straight-line trend along the video."""
    t = np.arange(len(scores), dtype=np.float64)
    return np.asarray(scores - np.polyval(np.polyfit(t, scores, 1), t), np.float64)
