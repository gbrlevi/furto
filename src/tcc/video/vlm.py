"""Zero-shot clip scores from a video-language model: the margin of "Yes" over "No"."""

import numpy as np
import numpy.typing as npt

from tcc.video.clips import clip_starts


def clip_layout(
    n_frames: int, fps: float, sample_fps: float, frames_per_clip: int, hop_seconds: float
) -> tuple[int, int, int, npt.NDArray[np.int64]]:
    """(step, span, hop, starts): a clip takes `frames_per_clip` frames every `step` frames
    (about `sample_fps`), covers `span` frames and a new one starts every `hop` frames."""
    step = max(round(fps / sample_fps), 1)
    span = (frames_per_clip - 1) * step + 1
    hop = max(round(hop_seconds * fps), 1)
    return step, span, hop, clip_starts(n_frames, span, hop)


def yes_no_margin(
    hidden: npt.NDArray[np.float32],
    yes_row: npt.NDArray[np.float32],
    no_row: npt.NDArray[np.float32],
) -> float:
    """Logit of "Yes" minus logit of "No" from the last hidden state, in float64."""
    h = hidden.astype(np.float64)
    return float(h @ yes_row.astype(np.float64) - h @ no_row.astype(np.float64))
