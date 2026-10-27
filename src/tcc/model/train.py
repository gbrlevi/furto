"""Training loop with early stopping on the validation loss.

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from tcc.model.loss import weighted_error


@dataclass
class TrainResult:
    history: list[dict[str, float]] = field(default_factory=list)
    best_state: dict[str, torch.Tensor] = field(default_factory=dict)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False  # False: the epoch limit ended the training


@torch.no_grad()
def gradient_norm(model: torch.nn.Module) -> float:
    """L2 norm of all parameter gradients together (instrumentation only)."""
    norms = [torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms)).item()) if norms else 0.0


@torch.no_grad()
def window_scores(
    model: torch.nn.Module,
    x: npt.NDArray[np.float32],
    device: torch.device,
    loss_norm: str,
    batch_size: int,
) -> npt.NDArray[np.float32]:
    """Reconstruction error of each window (the original `inferir`)."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        b = torch.from_numpy(x[i : i + batch_size]).to(device)
        out.append(weighted_error(model(b), b, loss_norm).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


@torch.no_grad()
def mean_loss(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    loss_norm: str,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for inputs, target in loader:
        e = weighted_error(model(inputs.to(device)), target.to(device), loss_norm)
        e = e[~torch.isnan(e)]
        total += e.sum().item()
        n += len(e)
    return total / max(n, 1)


def train(
    model: torch.nn.Module,
    train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
    val_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
    cfg: dict[str, Any],  # Any: the `train` section of the YAML config
    loss_norm: str,
    device: torch.device,
    generator: torch.Generator,
    log: Callable[[dict[str, float]], None] = lambda _: None,
) -> TrainResult:
    """Adam, optional ReduceLROnPlateau and early stopping; keeps the best epoch's weights.

    Each history row also records the mean and largest gradient norm and the largest batch
    loss of the epoch (session 3, P16 item 11). They are measured, never acted upon.
    """
    train_loader = DataLoader(
        train_set,
        cfg["batch_size"],
        shuffle=True,
        drop_last=len(train_set) > cfg["batch_size"],  # type: ignore[arg-type]
        generator=generator,
    )
    val_loader = DataLoader(val_set, cfg["batch_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    plateau = cfg["lr_plateau"]
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=plateau["factor"], patience=plateau["patience"]
        )
        if plateau["enabled"]
        else None
    )
    result = TrainResult()
    stale = 0
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        grad_norms, batch_losses = [], []
        for inputs, target in train_loader:
            e = weighted_error(model(inputs.to(device)), target.to(device), loss_norm)
            e = e[~torch.isnan(e)]
            if len(e) == 0:
                continue
            loss = e.mean()
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            grad_norms.append(gradient_norm(model))
            optimizer.step()
            batch_losses.append(loss.item())
            total += batch_losses[-1] * len(e)
            n += len(e)
        val_loss = mean_loss(model, val_loader, device, loss_norm)
        if scheduler is not None:
            scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
            "grad_norm_max": max(grad_norms, default=float("nan")),
            "batch_loss_max": max(batch_losses, default=float("nan")),
            "seconds": time.perf_counter() - start,
        }
        result.history.append(row)
        log(row)
        if val_loss < result.best_val_loss - 1e-6:
            result.best_val_loss, result.best_epoch, stale = val_loss, epoch, 0
            result.best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg["patience"]:
            result.stopped_early = True
            break
    return result


def loss_spikes(
    val_losses: list[float], factor: float, recover_epochs: int, recover_factor: float
) -> list[dict[str, Any]]:
    """Epochs (1-based) whose validation loss is at least `factor` times the best of the
    previous epochs, each with the first of the next `recover_epochs` epochs back within
    `recover_factor` times that best (None if none is). Consecutive epochs above the
    factor count as one spike, at the first of them. Session 3, P16 item 11."""
    out: list[dict[str, Any]] = []
    previous = False
    for i in range(1, len(val_losses)):
        best = min(val_losses[:i])
        high = val_losses[i] >= factor * best
        if not high or previous:
            previous = high
            continue
        previous = True
        after = val_losses[i + 1 : i + 1 + recover_epochs]
        back = [j for j, v in enumerate(after) if v <= recover_factor * best]
        out.append(
            {
                "epoch": i + 1,
                "val_loss": val_losses[i],
                "best_before": best,
                "recovered_epoch": i + 2 + back[0] if back else None,
            }
        )
    return out
