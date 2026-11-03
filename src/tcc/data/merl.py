"""MERL Shopping Dataset labels (Singh et al., CVPR 2016): one Matlab file per video with a
cell array `tlabs`; `tlabs{k}` lists [start, end] frames (1-based, 30 fps) of action k:
1 reach to shelf, 2 retract from shelf, 3 hand in shelf, 4 inspect product, 5 inspect shelf.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat

ACTIONS = ("reach", "retract", "hand_in_shelf", "inspect_product", "inspect_shelf")
SOURCE_FPS = 30.0


def load_labels(path: Path) -> dict[str, npt.NDArray[np.int64]]:
    """Action name -> (K, 2) [start, end] frames, 0-based at 30 fps, inclusive."""
    cells = loadmat(str(path))["tlabs"].ravel()
    return {
        name: np.asarray(c, np.int64).reshape(-1, 2) - 1
        for name, c in zip(ACTIONS, cells, strict=True)
    }


def split_of(subject: int) -> str:
    """The split of the dataset ReadMe: subjects 1-20 train, 21-26 validation, 27-41 test."""
    return "train" if subject <= 20 else "val" if subject <= 26 else "test"


def to_fps(spans: npt.NDArray[np.int64], fps: float) -> npt.NDArray[np.int64]:
    """30 fps frame spans to another frame rate (start floored, end ceiled)."""
    s = spans.astype(np.float64) * fps / SOURCE_FPS
    return np.column_stack([np.floor(s[:, 0]), np.ceil(s[:, 1])]).astype(np.int64)


def merge(spans: npt.NDArray[np.int64], max_gap: int) -> list[tuple[int, int]]:
    """Union of spans, joining those at most `max_gap` frames apart."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(map(tuple, spans.tolist())):
        if out and a <= out[-1][1] + max_gap + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out
