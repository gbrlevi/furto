"""Spatio-temporal graph (ST-GCN) autoencoder for anomaly detection (ADR-0002).

Adaptado de luizvitord/poc_furto@9bc4a91 (Luiz Vitor Dantas Freitas).
"""

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from tcc.data.skeleton import V, normalized_adjacency


class STGCNBlock(nn.Module):
    """Graph convolution over the skeleton followed by a temporal convolution
    (Yan; Xiong; Lin, 2018)."""

    A: torch.Tensor

    def __init__(
        self,
        c_in: int,
        c_out: int,
        adjacency: torch.Tensor,
        k_t: int = 9,
        stride: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("A", adjacency)
        self.importance = nn.Parameter(torch.ones_like(adjacency))  # learned weight per edge
        self.gcn = nn.Conv2d(c_in, c_out, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.tcn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (k_t, 1), (stride, 1), ((k_t - 1) // 2, 0)),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
        )
        self.res: nn.Module | None
        if not residual:
            self.res = None
        elif c_in == c_out and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, C, T, V)
        r = 0 if self.res is None else self.res(x)
        y = self.gcn(x)
        y = torch.einsum("nctv,vw->nctw", y, self.A * self.importance)
        y = self.tcn(self.bn(y))
        return F.relu(y + r)


class STGCNAutoencoder(nn.Module):
    """Takes (N, 3, T, 17) with x, y, confidence and reconstructs (N, 2, T, 17) with x, y."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 64),
        latent: int = 16,
        k_t: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        a = torch.tensor(normalized_adjacency())
        c1, c2, c3 = channels
        self.data_bn = nn.BatchNorm1d(3 * V)
        self.encoder = nn.ModuleList(
            [
                STGCNBlock(3, c1, a, k_t, 1, dropout, residual=False),
                STGCNBlock(c1, c2, a, k_t, 2, dropout),
                STGCNBlock(c2, c3, a, k_t, 2, dropout),
            ]
        )
        self.to_latent = nn.Conv2d(c3, latent, 1)
        self.from_latent = nn.Conv2d(latent, c3, 1)
        self.decoder = nn.ModuleList(
            [
                STGCNBlock(c3, c2, a, k_t, 1, dropout),
                STGCNBlock(c2, c1, a, k_t, 1, dropout),
                STGCNBlock(c1, c1, a, k_t, 1, dropout),
            ]
        )
        self.head = nn.Conv2d(c1, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = x.reshape(n, c, v, t).permute(0, 1, 3, 2)
        sizes = []
        for block in self.encoder:
            sizes.append(x.shape[2])
            x = block(x)
        x = F.relu(self.from_latent(self.to_latent(x)))
        # undo the temporal reductions in reverse order
        for block, t_target in zip(self.decoder, reversed(sizes), strict=True):
            if x.shape[2] != t_target:
                x = F.interpolate(x, size=(t_target, v), mode="nearest")
            x = block(x)
        out: torch.Tensor = self.head(x)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


_POC_PREFIXES = {
    "bn_dados.": "data_bn.",
    "enc.": "encoder.",
    "dec.": "decoder.",
    "para_latente.": "to_latent.",
    "de_latente.": "from_latent.",
    "saida.": "head.",
}


def state_dict_from_poc(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a state dict of the original `AutoencoderSTGCN` to this module's names."""
    out = {}
    for key, value in state.items():
        new = key.replace(".importancia", ".importance")
        for old_prefix, new_prefix in _POC_PREFIXES.items():
            if new.startswith(old_prefix):
                new = new_prefix + new[len(old_prefix) :]
                break
        else:
            raise KeyError(f"unexpected PoC state key {key!r}")
        out[new] = value
    return out
