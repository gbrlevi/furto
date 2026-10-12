"""Seeding of every random source used in training.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> np.random.Generator:
    """Seed python, numpy and torch (CPU and CUDA); return a numpy Generator."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - global state, for third-party code
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
