"""Training windows with data augmentation (Relatório, treatment step 8).

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import math

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from tcc.data.skeleton import MIRROR_INDEX


def mirror(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Horizontal mirror of a (3, T, 17) window: swap left/right joints and negate x.

    Correct only with x as the horizontal axis (ADR-0010).
    """
    out = x[:, :, MIRROR_INDEX].copy()
    out[0] *= -1
    return out


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Returns (input, target). With augmentation the input also loses random joints, so
    the model learns to reconstruct under occlusion (ADR-0007)."""

    def __init__(
        self,
        x: npt.NDArray[np.float32],
        rng: np.random.Generator,
        augment: bool = False,
        p_drop_joint: float = 0.1,
        rotation_deg: float = 10.0,
        scale: float = 0.1,
        noise: float = 0.01,
        p_mirror: float = 0.5,
        mask_target: bool = False,
    ) -> None:
        self.x = x
        self.rng = rng
        self.augment = augment
        self.p_drop_joint = p_drop_joint
        self.rotation_deg = rotation_deg
        self.scale = scale
        self.noise = noise
        self.p_mirror = p_mirror
        self.mask_target = mask_target

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[i].copy()  # (3, T, V)
        if self.augment:
            if self.rng.random() < self.p_mirror:
                x = mirror(x)
            valid = x[2] > 0
            th = math.radians(self.rng.uniform(-self.rotation_deg, self.rotation_deg))
            s = 1 + self.rng.uniform(-self.scale, self.scale)
            cx, cy = x[0].copy(), x[1].copy()
            x[0] = s * (math.cos(th) * cx - math.sin(th) * cy)
            x[1] = s * (math.sin(th) * cx + math.cos(th) * cy)
            x[:2] += self.rng.normal(0, self.noise, x[:2].shape).astype(np.float32)
            x[:2][:, ~valid] = 0
        target = x
        inputs = x
        if self.augment and self.p_drop_joint > 0:
            inputs = x.copy()
            dropped = self.rng.random(x.shape[2]) < self.p_drop_joint  # per joint
            inputs[:, :, dropped] = 0
            if self.mask_target:
                target = inputs
        return torch.from_numpy(inputs), torch.from_numpy(target)
