"""Confidence-weighted reconstruction error per window (ADR-0007).

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import torch

LOSS_NORMS = ("conf", "tv")


def weighted_error(
    reconstruction: torch.Tensor, target: torch.Tensor, loss_norm: str
) -> torch.Tensor:
    """Squared error per window, weighted by the confidence of each joint.

    reconstruction: (N, 2, T, V); target: (N, 3, T, V).
    - loss_norm="conf": divided by the sum of the window's confidences (ADR-0007, PoC);
    - loss_norm="tv": divided by T * V (Equation 2 of the Relatório).
    Windows without any confident point give NaN in both cases.
    """
    if loss_norm not in LOSS_NORMS:
        raise ValueError(f"loss_norm must be one of {LOSS_NORMS}")
    w = target[:, 2]
    e = ((reconstruction - target[:, :2]) ** 2).sum(1)
    den = w.sum((1, 2))
    num = (w * e).sum((1, 2))
    nan = torch.full_like(den, float("nan"))
    if loss_norm == "conf":
        return torch.where(den > 1e-6, num / den.clamp_min(1e-6), nan)
    t, v = w.shape[1], w.shape[2]
    return torch.where(den > 1e-6, num / (t * v), nan)
