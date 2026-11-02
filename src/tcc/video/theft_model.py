"""Clip classifiers on pretrained video features, and the within-video test against chance."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

Floats = npt.NDArray[np.floating[Any]]
Ints = npt.NDArray[np.int64]


def clip_labels(
    starts: Ints, frame_y: Ints, span: int, overlap: float
) -> npt.NDArray[np.int64]:  # fmt: skip
    """A clip is theft if at least `overlap` of its frames are."""
    total = np.concatenate([[0], np.cumsum(frame_y)])
    inside = total[starts + span] - total[starts]
    return np.asarray(inside >= overlap * span, np.int64)


def fit_linear(x: Floats, y: Ints, c: float) -> Any:  # sklearn pipelines have no useful type
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    )
    return model.fit(x, y)


def predict_linear(model: Any, x: Floats) -> npt.NDArray[np.float64]:
    return np.asarray(model.decision_function(x), np.float64)


class MILNet(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.net(x)).squeeze(-1)


def fit_mil(
    strong_x: Floats,
    strong_y: Ints,
    bags: Sequence[Floats],
    params: Mapping[str, Any],
    seed: int,
    device: str,
) -> MILNet:
    """Annotated clips with their labels, plus each unannotated theft video as a bag whose
    top-scoring clips count as theft in each epoch (multiple-instance learning)."""
    torch.manual_seed(seed)
    net = MILNet(strong_x.shape[1], params["hidden"], params["dropout"]).to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    sx = torch.as_tensor(strong_x, dtype=torch.float32, device=device)
    sy = torch.as_tensor(strong_y, dtype=torch.float32, device=device)
    tb = [torch.as_tensor(b, dtype=torch.float32, device=device) for b in bags if len(b)]
    for _ in range(params["epochs"]):
        net.eval()
        with torch.no_grad():
            top = [
                b[net(b).topk(max(1, int(params["topk_fraction"] * len(b)))).indices] for b in tb
            ]
        x = torch.cat([sx, *top])
        y = torch.cat([sy, torch.ones(sum(len(t) for t in top), device=device)])
        weight = (y == 0).sum() / (y == 1).sum().clamp(min=1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        net.train()
        order = torch.randperm(len(x), device=device)
        for i in range(0, len(order), params["batch"]):
            batch = order[i : i + params["batch"]]
            optimizer.zero_grad()
            loss_fn(net(x[batch]), y[batch]).backward()
            optimizer.step()
    return net.eval()


@torch.no_grad()
def predict_mil(net: MILNet, x: Floats, device: str) -> npt.NDArray[np.float64]:
    if not len(x):
        return np.zeros(0)
    return np.asarray(net(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu(), np.float64)


def within_video_auc(
    labels: Sequence[Ints], scores: Sequence[npt.NDArray[np.float64]], draws: int, seed: int
) -> dict[str, float]:
    """Mean AUC inside each video that has both classes, against the same mean with the labels
    circularly shifted inside each video (which keeps their temporal structure)."""
    rng = np.random.default_rng(seed)
    use = [(y, s) for y, s in zip(labels, scores, strict=True) if 0 < y.sum() < len(y)]
    observed = float(np.mean([roc_auc_score(y, s) for y, s in use]))
    null = []
    for _ in range(draws):
        values = []
        for y, s in use:
            shifted = np.roll(y, rng.integers(1, len(y)))
            if 0 < shifted.sum() < len(y):
                values.append(roc_auc_score(shifted, s))
        null.append(np.mean(values))
    low, high = np.quantile(null, [0.025, 0.975])
    return {
        "auc": observed,
        "videos": float(len(use)),
        "null_low": float(low),
        "null_high": float(high),
        "p": float((1 + np.sum(np.asarray(null) >= observed)) / (1 + draws)),
    }
