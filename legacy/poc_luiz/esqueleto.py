"""Definição do esqueleto COCO-17 e do grafo usado pela ST-GCN."""
import numpy as np

V = 17  # número de articulações

NOMES = [
    "nariz", "olho_esq", "olho_dir", "orelha_esq", "orelha_dir",
    "ombro_esq", "ombro_dir", "cotovelo_esq", "cotovelo_dir",
    "punho_esq", "punho_dir", "quadril_esq", "quadril_dir",
    "joelho_esq", "joelho_dir", "tornozelo_esq", "tornozelo_dir",
]

# 19 arestas espaciais (ossos) do padrão COCO
OSSOS = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# pares esquerda/direita, usados no espelhamento horizontal
PARES_ESPELHO = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

INDICE_ESPELHO = list(range(V))
for _a, _b in PARES_ESPELHO:
    INDICE_ESPELHO[_a], INDICE_ESPELHO[_b] = _b, _a

QUADRIS = [11, 12]
OMBROS = [5, 6]


def adjacencia_normalizada() -> np.ndarray:
    """Matriz  D^(-1/2) (A + I) D^(-1/2)  do esqueleto (Equação 1 da metodologia)."""
    a = np.eye(V, dtype=np.float32)
    for i, j in OSSOS:
        a[i, j] = a[j, i] = 1.0
    d = a.sum(1) ** -0.5
    return (d[:, None] * a * d[None, :]).astype(np.float32)
