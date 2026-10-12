"""Avalia um modelo treinado no conjunto de teste.

Uso básico:
    python avaliar.py --execucao runs/base

Calcula:
  * métricas independentes de limiar (por quadro): AUC-ROC, AUC-PR e EER;
  * no limiar operacional (escolhido só na validação, que contém apenas normais):
    precisão, revocação, F1, especificidade e H-PRS;
  * métricas operacionais: furtos detectados por evento e alertas por hora;
  * custo: parâmetros e latência por janela.
Resultados em runs/<nome>/resultados.json (ou resultados_<sufixo>.json).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from dados import Config, carregar_conjunto, gerar_janelas, remover_pontos, trajetorias
from modelo import AutoencoderSTGCN, contar_parametros, erro_ponderado, escolher_dispositivo

REFERENCIAS_POSELIFT = {  # AUC-ROC (%) publicadas no teste oficial do PoseLift
    "STG-NF": 67.46, "TSGAD": 63.35, "GEPC": 60.61, "Shopformer": 69.15,
}


# --------------------------------------------------------------------------- #
# Pontuação
# --------------------------------------------------------------------------- #
@torch.no_grad()
def inferir(modelo, x, disp, lote=1024):
    modelo.eval()
    saida = []
    for i in range(0, len(x), lote):
        b = torch.from_numpy(x[i:i + lote]).to(disp)
        saida.append(erro_ponderado(modelo(b), b).cpu().numpy())
    return np.concatenate(saida) if saida else np.zeros(0)


def pontuar_video(video, modelo, cfg, disp, frac_remover=0.0, rng=None):
    """Score bruto por quadro: maior score entre as janelas e pessoas que cobrem o quadro.
    Quadros sem nenhuma pessoa ficam NaN."""
    trajs = trajetorias(video, cfg)
    if frac_remover > 0:
        trajs = remover_pontos(trajs, frac_remover, rng)
    x, meta = gerar_janelas(trajs, cfg, cfg.passo_teste)
    s = np.full(video.n_quadros, np.nan)
    if len(x) == 0:
        return s
    scores = inferir(modelo, x, disp)
    for sc, q0 in zip(scores, meta[:, 2]):
        if np.isfinite(sc):
            fim = min(q0 + cfg.T, video.n_quadros)
            s[q0:fim] = np.fmax(s[q0:fim], sc)
    return s


def pos_processar(brutos, cameras, piso, sigma, stats_camera=None):
    """Normalização opcional por câmera, preenchimento dos quadros vazios e suavização."""
    out = []
    for s, cam in zip(brutos, cameras):
        s = s.copy()
        if stats_camera is not None:
            mu, dp = stats_camera.get(cam, stats_camera["_global"])
            s = (s - mu) / dp
        s[~np.isfinite(s)] = piso
        if sigma > 0:
            s = gaussian_filter1d(s, sigma)
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Alertas e eventos
# --------------------------------------------------------------------------- #
def intervalos(binario):
    """Lista de (início, fim exclusivo) dos trechos True."""
    b = np.concatenate([[0], binario.astype(np.int8), [0]])
    d = np.diff(b)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def alertas(score, limiar, juntar):
    """Trechos acima do limiar, unindo os que estão a menos de `juntar` quadros."""
    ints = intervalos(score >= limiar)
    unidos = []
    for a, b in ints:
        if unidos and a - unidos[-1][1] < juntar:
            unidos[-1] = (unidos[-1][0], b)
        else:
            unidos.append((a, b))
    return unidos


def alertas_por_hora(scores, limiar, juntar, fps):
    n_alertas = sum(len(alertas(s, limiar, juntar)) for s in scores)
    horas = sum(len(s) for s in scores) / fps / 3600
    return n_alertas / horas if horas > 0 else float("nan")


def presenca(video):
    """Máscara booleana dos quadros em que há pelo menos uma pessoa."""
    m = np.zeros(video.n_quadros, bool)
    for q, _ in video.poses.values():
        m[q] = True
    return m


def limiar_da_validacao(scores_val, juntar, fps, quantil=None, alvo_hora=None, amostra=None):
    """Limiar a partir da validação (só normais). `amostra` = scores usados no quantil."""
    todos = np.concatenate(scores_val) if amostra is None else amostra
    if alvo_hora is None:
        return float(np.quantile(todos, quantil)), f"quantil {quantil} da validação"
    cands = np.unique(np.quantile(todos, np.linspace(0.5, 0.99999, 400)))
    for thr in cands:
        if alertas_por_hora(scores_val, thr, juntar, fps) <= alvo_hora:
            return float(thr), f"<= {alvo_hora} alertas/hora na validação"
    return float(cands[-1]), f"<= {alvo_hora} alertas/hora (não atingido; usado o maior limiar)"


def avaliar_eventos(scores, rotulos, limiar, juntar, tolerancia):
    detectados = total = alertas_total = falsos = 0
    for s, y in zip(scores, rotulos):
        eventos = intervalos(y == 1)
        alts = alertas(s, limiar, juntar)
        total += len(eventos)
        alertas_total += len(alts)
        for a, b in eventos:
            if any(x < b + tolerancia and z > a - tolerancia for x, z in alts):
                detectados += 1
        for x, z in alts:
            if not any(x < b + tolerancia and z > a - tolerancia for a, b in eventos):
                falsos += 1
    return {"eventos": total, "eventos_detectados": detectados,
            "revocacao_eventos": detectados / total if total else float("nan"),
            "alertas": alertas_total, "alertas_falsos": falsos}


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def metricas_limiar(y, s, limiar):
    p = s >= limiar
    vp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum()); vn = int((~p & (y == 0)).sum())
    prec = vp / (vp + fp) if vp + fp else 0.0
    rev = vp / (vp + fn) if vp + fn else 0.0
    esp = vn / (vn + fp) if vn + fp else 0.0
    f1 = 2 * prec * rev / (prec + rev) if prec + rev else 0.0
    hprs = 3 / (1 / prec + 1 / rev + 1 / esp) if min(prec, rev, esp) > 0 else 0.0
    return {"precisao": prec, "revocacao": rev, "especificidade": esp, "f1": f1, "h_prs": hprs,
            "matriz_confusao": {"VP": vp, "FP": fp, "FN": fn, "VN": vn}}


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def latencia(modelo, cfg, disp, repeticoes=200):
    x = torch.randn(1, 3, cfg.T, 17, device=disp)
    modelo.eval()
    tempos = []
    with torch.no_grad():
        for _ in range(20):
            modelo(x)
        for _ in range(repeticoes):
            if disp.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            modelo(x)
            if disp.type == "cuda":
                torch.cuda.synchronize()
            tempos.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(tempos)), float(np.percentile(tempos, 95))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--execucao", required=True, help="pasta da execução, ex.: runs/base")
    ap.add_argument("--dados", help="por padrão, a mesma pasta usada no treino")
    ap.add_argument("--divisao", help="por padrão, a mesma do treino")
    ap.add_argument("--sigma", type=float, default=5.0, help="suavização gaussiana (quadros)")
    ap.add_argument("--quantil", type=float, default=0.95,
                    help="limiar = este quantil dos scores normais da validação")
    ap.add_argument("--alertas-hora", type=float,
                    help="alternativa ao quantil: maior taxa de alertas tolerada pelo operador")
    ap.add_argument("--juntar-s", type=float, default=2.0,
                    help="alertas separados por menos que isso (segundos) viram um só")
    ap.add_argument("--tolerancia-s", type=float, default=1.0,
                    help="folga (segundos) ao casar alertas com eventos de furto")
    ap.add_argument("--normalizar-camera", action="store_true",
                    help="padroniza os scores por câmera com estatísticas da validação")
    ap.add_argument("--remover-pontos", type=float, default=0.0,
                    help="oclusão simulada no teste: fração de articulações removidas")
    ap.add_argument("--quadros", default="com_pessoa", choices=["com_pessoa", "todos"],
                    help="quadros usados nas métricas por quadro (padrão: só com pessoa, "
                         "como no benchmark do PoseLift)")
    ap.add_argument("--sufixo", default="", help="sufixo do arquivo de resultados")
    ap.add_argument("--graficos", action="store_true", help="salva o score de cada vídeo de teste")
    ap.add_argument("--dispositivo", default="auto")
    args = ap.parse_args()

    pasta = Path(args.execucao)
    ck = torch.load(pasta / "modelo.pt", map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    disp = escolher_dispositivo(args.dispositivo)
    m = ck["modelo"]
    modelo = AutoencoderSTGCN(tuple(m["canais"]), m["latente"], m["k_t"], m["dropout"])
    modelo.load_state_dict(ck["estado"])
    modelo.to(disp).eval()

    dados = args.dados or ck["dados"]
    divisao = args.divisao or ck.get("divisao")
    print(f"Modelo da época {ck['epoca']} | dados: {dados}")
    conj, _ = carregar_conjunto(dados, divisao, ck.get("ordem", "auto"),
                                conf_nan=ck.get("conf_nan", 1.0))
    nomes_val = set(ck["videos_validacao"])
    val = [v for v in conj["treino"] if v.nome in nomes_val]
    teste = conj["teste"]
    if not val or not teste:
        raise SystemExit("Validação ou teste vazios. Confira --dados e --divisao.")
    sem_rot = [v.nome for v in teste if v.rotulos is None]
    if sem_rot:
        raise SystemExit(f"Vídeos de teste sem rótulo: {sem_rot[:5]}")

    rng = np.random.default_rng(ck["semente"])
    print("Pontuando validação e teste...")
    bruto_val = [pontuar_video(v, modelo, cfg, disp) for v in val]
    bruto_te = [pontuar_video(v, modelo, cfg, disp, args.remover_pontos, rng) for v in teste]

    stats = None
    if args.normalizar_camera:
        stats = {}
        por_cam = {}
        for s, v in zip(bruto_val, val):
            por_cam.setdefault(v.camera, []).append(s[np.isfinite(s)])
        todos = np.concatenate([np.concatenate(x) for x in por_cam.values()])
        stats["_global"] = (todos.mean(), todos.std() + 1e-8)
        for cam, xs in por_cam.items():
            x = np.concatenate(xs)
            if len(x) > 100:
                stats[cam] = (x.mean(), x.std() + 1e-8)

    finitos = np.concatenate([s[np.isfinite(s)] for s in bruto_val])
    if stats is not None:
        finitos = (finitos - stats["_global"][0]) / stats["_global"][1]
    piso = float(finitos.min())
    s_val = pos_processar(bruto_val, [v.camera for v in val], piso, args.sigma, stats)
    s_te = pos_processar(bruto_te, [v.camera for v in teste], piso, args.sigma, stats)

    # Protocolo do benchmark: métricas por quadro só nos quadros com pelo menos uma pessoa
    # (as contagens publicadas do PoseLift seguem essa convenção). --quadros todos usa tudo.
    pres_te = [presenca(v) for v in teste]
    pres_val = [presenca(v) for v in val]
    if args.quadros == "todos":
        pres_te = [np.ones_like(p) for p in pres_te]
        pres_val = [np.ones_like(p) for p in pres_val]
    y = np.concatenate([v.rotulos[p] for v, p in zip(teste, pres_te)])
    s = np.concatenate([sc[p] for sc, p in zip(s_te, pres_te)])
    if y.min() == y.max():
        raise SystemExit("O teste tem uma só classe; não dá para calcular AUC.")

    juntar = int(round(args.juntar_s * cfg.fps))
    tol = int(round(args.tolerancia_s * cfg.fps))
    amostra_val = np.concatenate([sc[p] for sc, p in zip(s_val, pres_val)])
    limiar, criterio = limiar_da_validacao(s_val, juntar, cfg.fps, args.quantil, args.alertas_hora,
                                           amostra_val)

    res = {
        "auc_roc": float(roc_auc_score(y, s)),
        "auc_pr": float(average_precision_score(y, s)),
        "auc_pr_aleatorio": float(y.mean()),
        "eer": eer(y, s),
        "limiar": limiar,
        "criterio_limiar": criterio,
        **metricas_limiar(y, s, limiar),
        **avaliar_eventos(s_te, [v.rotulos for v in teste], limiar, juntar, tol),
        "alertas_hora_validacao": alertas_por_hora(s_val, limiar, juntar, cfg.fps),
        "horas_validacao": sum(len(x) for x in s_val) / cfg.fps / 3600,
        "quadros_teste": int(len(y)),
        "quadros_teste_furto": int(y.sum()),
        "parametros": contar_parametros(modelo),
    }
    res["latencia_ms_cpu"], res["latencia_ms_cpu_p95"] = latencia(modelo.cpu(), cfg, torch.device("cpu"))
    modelo.to(disp)
    if disp.type != "cpu":
        res[f"latencia_ms_{disp.type}"], _ = latencia(modelo, cfg, disp)
    res["opcoes"] = {k: v for k, v in vars(args).items() if k not in ("execucao",)}

    sufixo = f"_{args.sufixo}" if args.sufixo else ""
    with open(pasta / f"resultados{sufixo}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(pasta / f"pontuacoes_teste{sufixo}.npz",
                        **{v.nome: sc for v, sc in zip(teste, s_te)},
                        **{f"rotulo__{v.nome}": v.rotulos for v in teste})

    print("\n=== Teste ===")
    print(f"Quadros avaliados ({args.quadros}): {res['quadros_teste']:,} | "
          f"de furto: {res['quadros_teste_furto']:,}")
    print(f"AUC-ROC {100 * res['auc_roc']:.2f}% | AUC-PR {100 * res['auc_pr']:.2f}% "
          f"(aleatório: {100 * res['auc_pr_aleatorio']:.2f}%) | EER {res['eer']:.3f}")
    print(f"Limiar {limiar:.4g} ({criterio})")
    print(f"Precisão {res['precisao']:.3f} | Revocação {res['revocacao']:.3f} | F1 {res['f1']:.3f} | "
          f"Especificidade {res['especificidade']:.3f} | H-PRS {res['h_prs']:.3f}")
    print(f"Furtos detectados: {res['eventos_detectados']}/{res['eventos']} "
          f"({100 * res['revocacao_eventos']:.1f}%) | alertas no teste: {res['alertas']} "
          f"(falsos: {res['alertas_falsos']})")
    print(f"Alertas por hora na validação (só normais): {res['alertas_hora_validacao']:.1f} "
          f"em {res['horas_validacao']:.2f} h")
    print(f"Modelo: {res['parametros']:,} parâmetros | CPU: {res['latencia_ms_cpu']:.2f} ms/janela "
          f"(p95 {res['latencia_ms_cpu_p95']:.2f})")
    print("Referência PoseLift (AUC-ROC): " +
          " | ".join(f"{k} {v:.2f}%" for k, v in REFERENCIAS_POSELIFT.items()))
    print(f"\nResultados em {pasta / f'resultados{sufixo}.json'}")

    if args.graficos:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gdir = pasta / f"graficos{sufixo}"
        gdir.mkdir(exist_ok=True)
        for v, sc in zip(teste, s_te):
            fig, ax = plt.subplots(figsize=(9, 2.6))
            t = np.arange(len(sc)) / cfg.fps
            for a, b in intervalos(v.rotulos == 1):
                ax.axvspan(a / cfg.fps, b / cfg.fps, color="crimson", alpha=0.18, lw=0)
            ax.plot(t, sc, color="#2a6f97", lw=1.2, label="score")
            ax.axhline(limiar, color="#444", ls="--", lw=0.8, label="limiar")
            ax.set_xlabel("segundos")
            ax.set_title(f"{v.nome}  (faixa vermelha = furto rotulado)", fontsize=9)
            ax.legend(fontsize=7, loc="upper left", frameon=False)
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            fig.savefig(gdir / f"{v.nome}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        print(f"Gráficos em {gdir}")


if __name__ == "__main__":
    main()
