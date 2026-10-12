"""Autoencoder espaço-temporal em grafo (ST-GCN) para detecção de anomalias."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from esqueleto import V, adjacencia_normalizada


class BlocoSTGCN(nn.Module):
    """Convolução espacial no grafo do esqueleto + convolução temporal (Yan; Xiong; Lin, 2018)."""

    def __init__(self, c_in, c_out, A, k_t=9, passo=1, dropout=0.1, residual=True):
        super().__init__()
        self.register_buffer("A", A)
        self.importancia = nn.Parameter(torch.ones_like(A))  # peso aprendido por aresta
        self.gcn = nn.Conv2d(c_in, c_out, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.tcn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (k_t, 1), (passo, 1), ((k_t - 1) // 2, 0)),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
        )
        if not residual:
            self.res = None
        elif c_in == c_out and passo == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(c_in, c_out, 1, (passo, 1)), nn.BatchNorm2d(c_out))

    def forward(self, x):  # x: (N, C, T, V)
        r = 0 if self.res is None else self.res(x)
        y = self.gcn(x)
        y = torch.einsum("nctv,vw->nctw", y, self.A * self.importancia)
        y = self.tcn(self.bn(y))
        return F.relu(y + r)


class AutoencoderSTGCN(nn.Module):
    """Recebe (N, 3, T, 17) com x, y, confiança e reconstrói (N, 2, T, 17) com x, y."""

    def __init__(self, canais=(32, 64, 64), latente=16, k_t=9, dropout=0.1):
        super().__init__()
        A = torch.tensor(adjacencia_normalizada())
        c1, c2, c3 = canais
        self.bn_dados = nn.BatchNorm1d(3 * V)
        self.enc = nn.ModuleList([
            BlocoSTGCN(3, c1, A, k_t, 1, dropout, residual=False),
            BlocoSTGCN(c1, c2, A, k_t, 2, dropout),
            BlocoSTGCN(c2, c3, A, k_t, 2, dropout),
        ])
        self.para_latente = nn.Conv2d(c3, latente, 1)
        self.de_latente = nn.Conv2d(latente, c3, 1)
        self.dec = nn.ModuleList([
            BlocoSTGCN(c3, c2, A, k_t, 1, dropout),
            BlocoSTGCN(c2, c1, A, k_t, 1, dropout),
            BlocoSTGCN(c1, c1, A, k_t, 1, dropout),
        ])
        self.saida = nn.Conv2d(c1, 2, 1)
        self.config = {"canais": list(canais), "latente": latente, "k_t": k_t, "dropout": dropout}

    def forward(self, x):
        n, c, t, v = x.shape
        x = self.bn_dados(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = x.reshape(n, c, v, t).permute(0, 1, 3, 2)
        tamanhos = []
        for bloco in self.enc:
            tamanhos.append(x.shape[2])
            x = bloco(x)
        x = F.relu(self.de_latente(self.para_latente(x)))
        # desfaz as reduções temporais na ordem inversa
        for bloco, t_alvo in zip(self.dec, [tamanhos[2], tamanhos[1], tamanhos[0]]):
            if x.shape[2] != t_alvo:
                x = F.interpolate(x, size=(t_alvo, v), mode="nearest")
            x = bloco(x)
        return self.saida(x)


def erro_ponderado(rec, alvo):
    """Erro quadrático por janela, ponderado pela confiança de cada articulação.

    rec: (N, 2, T, V); alvo: (N, 3, T, V). Janelas sem nenhum ponto válido -> NaN.
    """
    w = alvo[:, 2]
    e = ((rec - alvo[:, :2]) ** 2).sum(1)
    den = w.sum((1, 2))
    num = (w * e).sum((1, 2))
    return torch.where(den > 1e-6, num / den.clamp_min(1e-6), torch.full_like(den, float("nan")))


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def escolher_dispositivo(pedido="auto"):
    if pedido != "auto":
        return torch.device(pedido)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
