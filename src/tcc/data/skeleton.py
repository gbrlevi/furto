"""COCO-17 skeleton and the graph used by the ST-GCN.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import numpy as np
import numpy.typing as npt

V = 17  # number of joints

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]  # fmt: skip

# 19 spatial edges (bones) of the COCO layout
EDGES = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]  # fmt: skip

# left/right pairs, used by horizontal mirroring
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

MIRROR_INDEX = list(range(V))
for _a, _b in MIRROR_PAIRS:
    MIRROR_INDEX[_a], MIRROR_INDEX[_b] = _b, _a

HIPS = [11, 12]
SHOULDERS = [5, 6]


def normalized_adjacency() -> npt.NDArray[np.float32]:
    """D^(-1/2) (A + I) D^(-1/2) of the skeleton (Equation 1 of the methodology)."""
    a = np.eye(V, dtype=np.float32)
    for i, j in EDGES:
        a[i, j] = a[j, i] = 1.0
    d: npt.NDArray[np.float32] = a.sum(1) ** -0.5
    return (d[:, None] * a * d[None, :]).astype(np.float32)
