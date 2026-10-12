"""Treina o autoencoder ST-GCN só com comportamento normal.

Uso básico:
    python treinar.py --dados caminho/para/PoseLift --nome base

Saída em runs/<nome>/: modelo.pt (melhor época), historico.csv e config.json.
"""
import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dados import Config, carregar_conjunto, janelas_de_videos
from esqueleto import INDICE_ESPELHO
from modelo import AutoencoderSTGCN, contar_parametros, erro_ponderado, escolher_dispositivo


class JanelasDataset(Dataset):
    """Devolve (entrada, alvo). Com aumento, a entrada também perde articulações ao acaso
    (autoencoder com remoção de ruído): o modelo aprende a reconstruir mesmo com oclusão."""

    def __init__(self, x, aumentar=False, p_remover=0.1, rot_graus=10, escala=0.1, ruido=0.01):
        self.x = x
        self.aumentar = aumentar
        self.p_remover, self.rot, self.escala, self.ruido = p_remover, rot_graus, escala, ruido

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i].copy()  # (3, T, V)
        if self.aumentar:
            valido = x[2] > 0
            if random.random() < 0.5:  # espelhamento com troca esquerda/direita
                x = x[:, :, INDICE_ESPELHO]
                x[0] *= -1
                valido = x[2] > 0
            th = math.radians(random.uniform(-self.rot, self.rot))
            s = 1 + random.uniform(-self.escala, self.escala)
            cx, cy = x[0].copy(), x[1].copy()
            x[0] = s * (math.cos(th) * cx - math.sin(th) * cy)
            x[1] = s * (math.sin(th) * cx + math.cos(th) * cy)
            x[:2] += np.random.normal(0, self.ruido, x[:2].shape).astype(np.float32)
            x[:2][:, ~valido] = 0
        alvo = x
        entrada = x
        if self.aumentar and self.p_remover > 0:
            entrada = x.copy()
            mascara = np.random.random(x.shape[2]) < self.p_remover  # por articulação
            entrada[:, :, mascara] = 0
        return torch.from_numpy(entrada), torch.from_numpy(alvo)


def fixar_semente(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def dividir_por_video(videos, frac, semente):
    rng = np.random.default_rng(semente)
    idx = rng.permutation(len(videos))
    n_val = max(1, int(round(frac * len(videos))))
    val = {videos[i].nome for i in idx[:n_val]}
    return [v for v in videos if v.nome not in val], [v for v in videos if v.nome in val]


@torch.no_grad()
def perda_media(modelo, carregador, disp):
    modelo.eval()
    soma, n = 0.0, 0
    for ent, alvo in carregador:
        ent, alvo = ent.to(disp), alvo.to(disp)
        e = erro_ponderado(modelo(ent), alvo)
        e = e[~torch.isnan(e)]
        soma += e.sum().item()
        n += len(e)
    return soma / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dados", required=True)
    ap.add_argument("--divisao", help="arquivo com a divisão treino/teste (opcional)")
    ap.add_argument("--ordem", default="auto", choices=["auto", "quadro-pessoa", "pessoa-quadro"])
    ap.add_argument("--nome", default="base", help="nome da execução (pasta em runs/)")
    ap.add_argument("--saida", default="runs")
    # dados
    ap.add_argument("--T", type=int, default=24, help="quadros por janela")
    ap.add_argument("--passo-treino", type=int, default=6)
    ap.add_argument("--centro", default="quadro", choices=["quadro", "janela"])
    ap.add_argument("--conf-min", type=float, default=0.1)
    ap.add_argument("--conf-nan", type=float, default=1.0,
                    help="confiança usada quando ela vem NaN com x/y válidos (0 = descartar)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--frac-validacao", type=float, default=0.15)
    ap.add_argument("--max-janelas", type=int, default=0,
                    help="limita as janelas de treino (útil no RetailS); 0 = sem limite")
    # modelo
    ap.add_argument("--canais", default="32,64,64")
    ap.add_argument("--latente", type=int, default=16)
    ap.add_argument("--k-t", type=int, default=9, help="tamanho do filtro temporal")
    ap.add_argument("--dropout", type=float, default=0.1)
    # treino
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--paciencia", type=int, default=8, help="parada antecipada")
    ap.add_argument("--lote", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--sem-aumento", action="store_true")
    ap.add_argument("--p-remover", type=float, default=0.1, help="oclusão simulada no treino")
    ap.add_argument("--dispositivo", default="auto", help="auto, cpu, cuda ou mps")
    ap.add_argument("--trabalhadores", type=int, default=0, help="processos do DataLoader")
    args = ap.parse_args()

    fixar_semente(args.semente)
    disp = escolher_dispositivo(args.dispositivo)
    pasta = Path(args.saida) / args.nome
    pasta.mkdir(parents=True, exist_ok=True)
    cfg = Config(T=args.T, passo_treino=args.passo_treino, centro=args.centro,
                 conf_min=args.conf_min, fps=args.fps)

    print(f"Dispositivo: {disp}")
    print("Carregando dados...")
    conj, _ = carregar_conjunto(args.dados, args.divisao, args.ordem, conf_nan=args.conf_nan)
    treino_v, val_v = dividir_por_video(conj["treino"], args.frac_validacao, args.semente)
    print(f"Vídeos: {len(treino_v)} treino | {len(val_v)} validação | {len(conj['teste'])} teste")

    x_tr, _ = janelas_de_videos(treino_v, cfg, cfg.passo_treino)
    x_va, _ = janelas_de_videos(val_v, cfg, cfg.passo_treino)
    if args.max_janelas and len(x_tr) > args.max_janelas:
        sel = np.random.default_rng(args.semente).choice(len(x_tr), args.max_janelas, replace=False)
        x_tr = x_tr[np.sort(sel)]
    print(f"Janelas: {len(x_tr):,} treino | {len(x_va):,} validação (T={cfg.T})")
    if len(x_tr) == 0 or len(x_va) == 0:
        raise SystemExit("Nenhuma janela gerada. Rode inspecionar.py para conferir os dados.")

    ds_tr = JanelasDataset(x_tr, aumentar=not args.sem_aumento, p_remover=args.p_remover)
    ds_va = JanelasDataset(x_va)
    dl_tr = DataLoader(ds_tr, args.lote, shuffle=True, num_workers=args.trabalhadores,
                       drop_last=len(ds_tr) > args.lote)
    dl_va = DataLoader(ds_va, args.lote, num_workers=args.trabalhadores)

    canais = tuple(int(c) for c in args.canais.split(","))
    modelo = AutoencoderSTGCN(canais, args.latente, args.k_t, args.dropout).to(disp)
    print(f"Parâmetros treináveis: {contar_parametros(modelo):,}")
    otim = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    agenda = torch.optim.lr_scheduler.ReduceLROnPlateau(otim, factor=0.5, patience=3)

    melhor, sem_melhora, hist = float("inf"), 0, []
    for ep in range(1, args.epocas + 1):
        t0 = time.time()
        modelo.train()
        soma, n = 0.0, 0
        for ent, alvo in dl_tr:
            ent, alvo = ent.to(disp), alvo.to(disp)
            e = erro_ponderado(modelo(ent), alvo)
            e = e[~torch.isnan(e)]
            if len(e) == 0:
                continue
            perda = e.mean()
            otim.zero_grad()
            perda.backward()
            otim.step()
            soma += perda.item() * len(e)
            n += len(e)
        p_tr = soma / max(n, 1)
        p_va = perda_media(modelo, dl_va, disp)
        agenda.step(p_va)
        hist.append({"epoca": ep, "perda_treino": p_tr, "perda_validacao": p_va,
                     "lr": otim.param_groups[0]["lr"], "segundos": time.time() - t0})
        marca = ""
        if p_va < melhor - 1e-6:
            melhor, sem_melhora, marca = p_va, 0, "  *melhor"
            torch.save({
                "estado": modelo.state_dict(),
                "modelo": modelo.config,
                "cfg": asdict(cfg),
                "videos_validacao": sorted(v.nome for v in val_v),
                "dados": str(Path(args.dados).resolve()),
                "divisao": args.divisao,
                "ordem": args.ordem,
                "conf_nan": args.conf_nan,
                "semente": args.semente,
                "epoca": ep,
                "perda_validacao": p_va,
            }, pasta / "modelo.pt")
        else:
            sem_melhora += 1
        print(f"época {ep:3d} | treino {p_tr:.5f} | validação {p_va:.5f} | "
              f"{hist[-1]['segundos']:.1f}s{marca}")
        if sem_melhora >= args.paciencia:
            print(f"Parada antecipada: {args.paciencia} épocas sem melhora.")
            break

    with open(pasta / "historico.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    with open(pasta / "config.json", "w", encoding="utf-8") as f:
        json.dump({"argumentos": vars(args), "cfg": asdict(cfg),
                   "parametros": contar_parametros(modelo), "melhor_perda_validacao": melhor},
                  f, indent=2, ensure_ascii=False)
    print(f"\nMelhor perda de validação: {melhor:.5f}. Modelo salvo em {pasta / 'modelo.pt'}")
    print(f"Próximo passo: python avaliar.py --execucao {pasta}")


if __name__ == "__main__":
    main()
