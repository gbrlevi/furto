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
