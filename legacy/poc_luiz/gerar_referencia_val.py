"""Pontuações de validação do run de referência do PoC original (sessão 2, tarefa B4).

Única exceção autorizada à proibição de importar o legado (docs/tcc/PORTE_POC_LUIZ.md).
Lê só os vídeos de validação gravados no checkpoint; nenhum vídeo de teste é aberto.

    python legacy/poc_luiz/gerar_referencia_val.py data/reference/poc_luiz_seed0

Salva val_ref.npz na mesma pasta, por vídeo de validação <v>:
    <v>__windows             janelas normalizadas (N, 3, T, 17), passo de avaliação
    <v>__window_meta         (N, 3) = [0, pid, quadro inicial]
    <v>__window_scores       erro ponderado por janela (N,)
    <v>__frame_scores_raw    escore bruto por quadro (pontuar_video), NaN sem pessoa
    <v>__frame_scores_post   pos_processar com piso = mínimo da validação e sigma = 5
"""
import sys
from pathlib import Path

import numpy as np
import torch

from avaliar import inferir, pontuar_video, pos_processar
from dados import Config, carregar_conjunto, gerar_janelas, trajetorias
from modelo import AutoencoderSTGCN, escolher_dispositivo

SIGMA_PADRAO = 5.0  # padrão de --sigma em avaliar.py


def main():
    pasta = Path(sys.argv[1])
    saida = pasta / "val_ref.npz"
    if saida.exists():
        raise SystemExit(f"{saida} já existe; a referência nunca é sobrescrita")
    ck = torch.load(pasta / "modelo.pt", map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    disp = escolher_dispositivo("auto")
    m = ck["modelo"]
    modelo = AutoencoderSTGCN(tuple(m["canais"]), m["latente"], m["k_t"], m["dropout"])
    modelo.load_state_dict(ck["estado"])
    modelo.to(disp).eval()

    nomes_val = set(ck["videos_validacao"])
    conj, _ = carregar_conjunto(ck["dados"], ck["divisao"], ck["ordem"], verbose=False,
                                somente=nomes_val, conf_nan=ck["conf_nan"])
    val = [v for v in conj["treino"] if v.nome in nomes_val]
    if conj["teste"] or sorted(v.nome for v in val) != sorted(nomes_val):
        raise SystemExit("conjunto de validação diferente do gravado no checkpoint")

    arrays = {}
    brutos = []
    for v in val:
        x, meta = gerar_janelas(trajetorias(v, cfg), cfg, cfg.passo_teste)
        arrays[f"{v.nome}__windows"] = x
        arrays[f"{v.nome}__window_meta"] = meta
        arrays[f"{v.nome}__window_scores"] = inferir(modelo, x, disp)
        bruto = pontuar_video(v, modelo, cfg, disp)
        arrays[f"{v.nome}__frame_scores_raw"] = bruto
        brutos.append(bruto)
    finitos = np.concatenate([s[np.isfinite(s)] for s in brutos])
    piso = float(finitos.min())
    for v, s in zip(val, pos_processar(brutos, [v.camera for v in val], piso, SIGMA_PADRAO)):
        arrays[f"{v.nome}__frame_scores_post"] = s

    np.savez_compressed(saida, videos=np.array(sorted(nomes_val)), piso=piso,
                        sigma=SIGMA_PADRAO, epoca=ck["epoca"], dispositivo=str(disp), **arrays)
    print(f"{len(val)} vídeos de validação em {saida}")


if __name__ == "__main__":
    main()
