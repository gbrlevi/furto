"""Gera um conjunto sintético pequeno, só para testar se o código roda.

Os números de desempenho obtidos com ele NÃO significam nada sobre o PoseLift.

    python ferramentas/gerar_dados_falsos.py --saida dados_falsos --formato pkl
    python ferramentas/gerar_dados_falsos.py --saida dados_falsos_json --formato pastas
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

# pose em pé, em pixels, com origem nos pés (y cresce para baixo)
BASE = np.array([
    [0, -170], [5, -175], [-5, -175], [10, -172], [-10, -172],
    [25, -140], [-25, -140], [30, -108], [-30, -108], [32, -78], [-32, -78],
    [15, -88], [-15, -88], [16, -45], [-16, -45], [17, 0], [-17, 0],
], dtype=np.float32)


def pessoa(n, rng, furto=None):
    """Sequência (n, 17, 3) de uma pessoa andando, olhando e pegando produtos.
    furto=(ini, fim): leva o punho direito ao quadril e o mantém lá."""
    escala = rng.uniform(0.8, 1.3)
    x0, y0 = rng.uniform(300, 1500), rng.uniform(700, 1000)
    vel = rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)
    fase = rng.uniform(0, 6.28)
    pega = set(rng.choice(n, size=max(1, n // 120), replace=False).tolist())
    out = np.zeros((n, 17, 3), np.float32)
    alcance = 0.0
    for t in range(n):
        p = BASE.copy()
        anda = 1.0 if (t // 60) % 3 != 2 else 0.1  # alterna andar e parar
        w = np.sin(0.35 * t + fase) * anda
        p[[13, 15], 0] += 10 * w; p[[14, 16], 0] -= 10 * w
        p[[7, 9], 0] -= 8 * w;    p[[8, 10], 0] += 8 * w
        if t in pega:
            alcance = 1.0
        alcance *= 0.93            # braço sobe até a prateleira e volta
        p[9] += alcance * np.array([35, -90])
        p[7] += alcance * np.array([18, -45])
        if furto and furto[0] <= t < furto[1]:
            k = min(1.0, (t - furto[0]) / 8, (furto[1] - t) / 8)
            p[10] = (1 - k) * p[10] + k * (p[12] + np.array([-4, 6]))
            p[8] = (1 - k) * p[8] + k * (p[6] + np.array([-14, 30]))
            p[:5, 0] += 6 * k      # cabeça inclina
        x = x0 + vel * t * anda
        out[t, :, :2] = p * escala + [x, y0]
        out[t, :, :2] += rng.normal(0, 1.5, (17, 2))
        out[t, :, 2] = rng.uniform(0.55, 0.98, 17)
        out[t, rng.random(17) < 0.03, 2] = 0.05  # pontos perdidos
    return out


def video(n, n_pessoas, rng, com_furto):
    pessoas, rot = {}, np.zeros(n, np.int64)
    for i in range(n_pessoas):
        ini, fim = sorted(rng.choice(n, 2, replace=False))
        ini, fim = int(max(0, ini - 40)), int(min(n, max(fim, ini + 80)))
        furto = None
        if com_furto and i == 0:
            a = rng.integers(ini + 20, fim - 40)
            furto = (int(a), int(a + rng.integers(20, 40)))
            rot[furto[0]:furto[1]] = 1
        seq = pessoa(fim - ini, rng, None if furto is None else (furto[0] - ini, furto[1] - ini))
        pessoas[i + 1] = (np.arange(ini, fim), seq)
    return pessoas, rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--formato", choices=["pkl", "pastas"], default="pkl")
    ap.add_argument("--treino", type=int, default=24)
    ap.add_argument("--teste", type=int, default=8)
    ap.add_argument("--semente", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.semente)
    raiz = Path(a.saida)

    for k in range(a.treino + a.teste):
        teste = k >= a.treino
        nome = f"C{k % 6 + 1}_{k:03d}"
        pessoas, rot = video(int(rng.integers(250, 450)), int(rng.integers(1, 3)), rng, teste)
        if a.formato == "pkl":
            quadros = {}
            for pid, (qs, seq) in pessoas.items():
                for q, kp in zip(qs, seq):
                    x, y = kp[:, 0], kp[:, 1]
                    quadros.setdefault(int(q), []).append({
                        "person_id": pid,
                        "bbox": [float(x.min()), float(y.min()),
                                 float(x.max() - x.min()), float(y.max() - y.min())],
                        "keypoints": kp})
            raiz.mkdir(parents=True, exist_ok=True)
            with open(raiz / f"{nome}.pkl", "wb") as f:
                pickle.dump(quadros, f)
            np.save(raiz / f"{nome}.npy", rot)
        else:
            sub = "test" if teste else "train"
            d = raiz / "pose" / sub
            d.mkdir(parents=True, exist_ok=True)
            js = {str(pid): {str(int(q)): {"keypoints": kp.ravel().round(2).tolist(), "scores": 0.9}
                             for q, kp in zip(qs, seq)} for pid, (qs, seq) in pessoas.items()}
            with open(d / f"{nome}_alphapose_tracked_person.json", "w") as f:
                json.dump(js, f)
            if teste:
                g = raiz / "gt" / "test_frame_mask"
                g.mkdir(parents=True, exist_ok=True)
                np.save(g / f"{nome}.npy", rot)
    print(f"Dados sintéticos ({a.formato}) em {raiz}")


if __name__ == "__main__":
    main()
