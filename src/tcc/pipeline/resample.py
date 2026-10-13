"""Frame-rate resampling by frame selection (Relatório: decode at 15 fps)."""

import numpy as np
import numpy.typing as npt


def frames_to_keep(source_fps: float, n_frames: int, target_fps: float) -> npt.NDArray[np.int64]:
    """Source frame indices sampled at `target_fps`: output frame k takes the source frame
    nearest to time k / target_fps. Never upsamples."""
    if target_fps <= 0 or source_fps <= 0:
        raise ValueError("frame rates must be positive")
    if target_fps > source_fps:
        raise ValueError(f"cannot upsample {source_fps} fps to {target_fps} fps")
    duration = n_frames / source_fps
    k = np.arange(int(np.floor(duration * target_fps + 1e-9)))
    idx = np.round(k * source_fps / target_fps).astype(np.int64)
    return idx[idx < n_frames]
