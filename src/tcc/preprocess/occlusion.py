"""Simulated occlusion: zero the confidence of a random fraction of joints.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import numpy as np

from tcc.preprocess.trajectories import Trajectory


def drop_joints(
    trajectories: list[Trajectory], fraction: float, rng: np.random.Generator
) -> list[Trajectory]:
    out = []
    for tid, frames, keypoints in trajectories:
        k = keypoints.copy()
        k[..., 2][rng.random(k.shape[:2]) < fraction] = 0.0
        out.append(Trajectory(tid, frames, k))
    return out
