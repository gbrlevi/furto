"""Confere a estrutura dos dados antes de treinar.

Uso:
    python inspecionar.py --dados caminho/para/PoseLift
    python inspecionar.py --dados caminho/para/PoseLift --desenhar
"""
import argparse
from collections import Counter

import numpy as np

from dados import (Config, carregar_conjunto, extrair_registros, ler_arquivo_pose,
                   localizar, trajetorias)
from esqueleto import NOMES, OSSOS

# valores publicados para a partição oficial do PoseLift
ESPERADO_POSELIFT = {"treino_normais": 53_353, "teste_normais": 2_221, "teste_furto": 1_500}


def arvore(obj, prof=0, max_prof=4, nome="raiz"):
    pre = "  " * prof
    tipo = type(obj).__name__
    if isinstance(obj, dict):
        chaves = list(obj.keys())
        print(f"{pre}{nome}: dict com {len(chaves)} chaves, ex.: {chaves[:4]}")
        if prof < max_prof and chaves:
            arvore(obj[chaves[0]], prof + 1, max_prof, f"[{chaves[0]!r}]")
    elif isinstance(obj, (list, tuple)):
        print(f"{pre}{nome}: {tipo} com {len(obj)} itens")
        if prof < max_prof and obj:
            arvore(obj[0], prof + 1, max_prof, "[0]")
    elif isinstance(obj, np.ndarray):
        print(f"{pre}{nome}: ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        texto = repr(obj)
        print(f"{pre}{nome}: {tipo} = {texto[:60]}")


def contagens(videos):
    """Quadros normais/de furto em três convenções de contagem."""
    c = {"todos": [0, 0], "com_pessoa": [0, 0], "pessoa_quadro": [0, 0]}
    for v in videos:
        y = v.rotulos if v.rotulos is not None else np.zeros(v.n_quadros, np.int64)
        presenca = np.zeros(v.n_quadros, np.int64)
        for q, _ in v.poses.values():
            presenca[q] += 1
        for nome, peso in [("todos", np.ones_like(y)), ("com_pessoa", presenca > 0),
                           ("pessoa_quadro", presenca)]:
            peso = peso.astype(np.int64)
            c[nome][0] += int((peso * (y == 0)).sum())
            c[nome][1] += int((peso * (y == 1)).sum())
    return c


def resumo(videos, rotulo):
    c = contagens(videos)
    pessoas = sum(len(v.poses) for v in videos)
    cams = Counter(v.camera for v in videos)
    n, f = c["todos"]
    print(f"  {rotulo}: {len(videos)} vídeos | {n + f:,} quadros | {n:,} normais | "
          f"{f:,} de furto | {pessoas} IDs de pessoa | câmeras: {dict(cams)}")
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dados", required=True, help="pasta com os dados baixados")
    ap.add_argument("--divisao", help="arquivo com a divisão treino/teste (opcional)")
    ap.add_argument("--ordem", default="auto", choices=["auto", "quadro-pessoa", "pessoa-quadro"])
    ap.add_argument("--desenhar", action="store_true", help="salva um esqueleto em esqueleto.png")
    ap.add_argument("--conf-nan", type=float, default=1.0,
                    help="confiança usada quando ela vem NaN com x/y válidos (padrão 1)")
    args = ap.parse_args()

    info = localizar(args.dados)
    print(f"\n1) Arquivos encontrados em {info['raiz']}")
    print(f"  formato detectado: {info['formato']}")
    print(f"  arquivos de pose: {len(info['poses'])} | arquivos de rótulo (.npy): {len(info['rotulos'])}")
    if info["formato"] == "pastas":
        c = Counter(info["divisoes"].values())
        print(f"  por pasta: treino={c.get('treino', 0)} teste={c.get('teste', 0)} "
              f"outros={c.get(None, 0)}")
    if not info["poses"]:
        print("  Nenhum .pkl/.json encontrado. Confira o caminho passado em --dados.")
        return

    exemplo = info["poses"][0]
    print(f"\n2) Estrutura de um arquivo de exemplo: {exemplo.name}")
    bruto = ler_arquivo_pose(exemplo)
    arvore(bruto)
    regs = extrair_registros(bruto, args.ordem)
    if not regs:
        print("  Não reconheci poses neste arquivo. Envie a saída acima para ajustar o leitor.")
        return
    qs = [r[0] for r in regs]
    pids = {r[1] for r in regs}
    kps = np.stack([r[2] for r in regs])
    print(f"  poses lidas: {len(regs)} | quadros {min(qs)}..{max(qs)} | IDs: {sorted(pids)[:10]}")
    with np.errstate(all="ignore"):
        print(f"  x: {np.nanmin(kps[..., 0]):.1f}..{np.nanmax(kps[..., 0]):.1f} | "
              f"y: {np.nanmin(kps[..., 1]):.1f}..{np.nanmax(kps[..., 1]):.1f}")
    conf = kps[..., 2]
    nan_c = ~np.isfinite(conf)
    nan_xy = ~np.isfinite(kps[..., :2]).all(-1)
    texto_conf = (f"{np.nanmin(conf):.2f}..{np.nanmax(conf):.2f}" if (~nan_c).any() else "sem valores")
    print(f"  confiança (valores definidos): {texto_conf} | NaN na confiança: "
          f"{100 * nan_c.mean():.1f}% dos pontos | NaN em x/y: {100 * nan_xy.mean():.1f}%")
    if nan_c.any():
        print(f"  -> pontos com confiança NaN e x/y válidos recebem confiança {args.conf_nan} "
              "(opção --conf-nan).")
    if np.nanmax(kps[..., 0]) <= 1.5:
        print("  [atenção] coordenadas parecem normalizadas (0-1), não em pixels. Tudo bem, "
              "a normalização por tronco funciona igual.")

    print("\n3) Conjunto completo")
    conj, rel = carregar_conjunto(args.dados, args.divisao, args.ordem, conf_nan=args.conf_nan)
    c_tr = resumo(conj["treino"], "treino")
    c_te = resumo(conj["teste"], "teste ")
    if rel["sem_rotulo"]:
        print(f"  vídeos sem arquivo de rótulo (tratados como normais): {len(rel['sem_rotulo'])}")
    offs = Counter(rel["offsets"].values())
    print(f"  deslocamento de quadros aplicado: {dict(offs)}")

    print("\n4) Comparação com a partição oficial do PoseLift (Tabela 1 da metodologia)")
    print("  A contagem publicada pode usar outra convenção; veja qual delas chega mais perto:")
    print(f"  {'convenção':<34}{'treino normais':>16}{'teste normais':>16}{'teste furto':>14}")
    nomes = {"todos": "todos os quadros do vídeo",
             "com_pessoa": "quadros com pelo menos 1 pessoa",
             "pessoa_quadro": "pessoa x quadro (soma por pessoa)"}
    for k, rotulo in nomes.items():
        print(f"  {rotulo:<34}{c_tr[k][0]:>16,}{c_te[k][0]:>16,}{c_te[k][1]:>14,}")
    e = ESPERADO_POSELIFT
    print(f"  {'publicado':<34}{e['treino_normais']:>16,}{e['teste_normais']:>16,}"
          f"{e['teste_furto']:>14,}")
    print("  Diferenças moderadas são esperadas: a versão atual do PoseLift no Drive tem\n"
          "  quantidade de vídeos diferente da descrita no artigo. O que importa é a divisão\n"
          "  vir das pastas train/test oficiais e o teste ter as duas classes.")

    cfg = Config()
    n_traj = sum(len(trajetorias(v, cfg)) for v in conj["treino"])
    print(f"\n5) Trajetórias de treino com pelo menos {cfg.T} quadros: {n_traj}")

    if args.desenhar:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        v = next(v for v in conj["treino"] + conj["teste"] if v.poses)
        pid, (q, k) = next(iter(v.poses.items()))
        p = k[len(k) // 2]
        fig, ax = plt.subplots(figsize=(4, 6))
        for a, b in OSSOS:
            if p[a, 2] > 0.1 and p[b, 2] > 0.1:
                ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], "-", color="teal")
        for i, (x, y, c) in enumerate(p):
            if c > 0.1:
                ax.plot(x, y, "o", color="crimson")
                ax.annotate(NOMES[i], (x, y), fontsize=6)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"{v.nome} | pessoa {pid} | quadro {q[len(q) // 2]}")
        fig.savefig("esqueleto.png", dpi=150, bbox_inches="tight")
        print("\nEsqueleto salvo em esqueleto.png. Confira se 'punho_esq' está mesmo no braço "
              "esquerdo da pessoa (lado direito da imagem, se ela estiver de frente).")


if __name__ == "__main__":
    main()
