"""Leitura do PoseLift / RetailS e pré-processamento das poses em janelas.

Aceita os dois formatos em que os dados costumam aparecer:

  A) Arquivos soltos (README do PoseLift):
       <camera>_<video>.pkl   -> dicionário {quadro: pessoas}
       <camera>_<video>.npy   -> rótulo 0/1 por quadro
     A divisão treino/teste vem de --divisao ou, na falta dela, da regra
     "vídeo com algum quadro de furto vai para o teste".

  B) Pastas no formato STG-NF / Shopformer:
       .../pose/train/*.json, .../pose/test/*.json
       .../gt/test_frame_mask/*.npy
     JSON no formato {pessoa: {quadro: {"keypoints": [...51 valores...]}}}.

O leitor é tolerante: reconhece dicionários quadro->pessoa, pessoa->quadro,
listas de registros e pontos como lista de 51 valores ou matriz 17x3.
Rode `python inspecionar.py --dados <pasta>` antes de treinar para conferir.
"""
from __future__ import annotations

import json
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from esqueleto import OMBROS, QUADRIS, V

EXT_POSE = {".pkl", ".pickle", ".json"}

CHAVES_KP = ("keypoints", "keypoint", "kps", "kpts", "pose", "joints", "skeleton")
CHAVES_ID = ("person_id", "personid", "pid", "track_id", "trackid", "id", "idx", "person")
CHAVES_QUADRO = ("frame", "frame_id", "frameid", "frame_idx", "frame_number", "image_id")
SUFIXOS_CONHECIDOS = ("_alphapose_tracked_person", "_tracked_person", "_poses", "_pose")


# --------------------------------------------------------------------------- #
# Configuração e estruturas
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    T: int = 24                # quadros por janela
    passo_treino: int = 6      # deslocamento entre janelas no treino
    passo_teste: int = 1       # no teste, toda posição recebe janela
    lacuna_max: int = 8        # lacunas até esse tamanho são interpoladas
    conf_min: float = 0.1      # confiança abaixo disso = ponto ausente
    centro: str = "quadro"     # "quadro" (metodologia) ou "janela"
    fps: float = 15.0


@dataclass
class Video:
    nome: str
    camera: str
    poses: dict                # pid -> (quadros int64 (n,), pontos float32 (n,17,3))
    rotulos: np.ndarray | None # 0/1 por quadro, ou None (assume tudo normal)
    n_quadros: int


# --------------------------------------------------------------------------- #
# Interpretação genérica dos arquivos de pose
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"-?\d+")


def _int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return int(v)
    if isinstance(v, str):
        m = _NUM.findall(v)
        return int(m[-1]) if m else None
    return None


def _tenta_kps(v):
    """Converte v em matriz 17x3 (x, y, confiança), ou devolve None."""
    if v is None or isinstance(v, (dict, str)) or np.isscalar(v):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = np.asarray(v, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if a.size == V * 3:
        return a.reshape(V, 3)
    if a.size == V * 2:  # sem confiança: assume 1
        return np.concatenate([a.reshape(V, 2), np.ones((V, 1), np.float32)], 1)
    return None


def _pessoa(item, pid_padrao=None):
    """Tenta ler uma pessoa: devolve (pid, pontos 17x3) ou None."""
    if isinstance(item, dict):
        low = {str(k).lower(): k for k in item}
        kp = None
        for c in CHAVES_KP:
            if c in low:
                kp = _tenta_kps(item[low[c]])
                if kp is not None:
                    break
        if kp is None:
            return None
        pid = None
        for c in CHAVES_ID:
            if c in low:
                pid = _int(item[low[c]])
                if pid is not None:
                    break
        return (pid if pid is not None else pid_padrao, kp)

    if isinstance(item, (list, tuple, np.ndarray)):
        kp = _tenta_kps(item)          # a própria sequência são os pontos
        if kp is not None:
            return (pid_padrao, kp)
        if isinstance(item, np.ndarray) or all(isinstance(e, dict) for e in item):
            return None
        pid = None                     # sequência mista, ex.: [pid, caixa, pontos]
        for e in item:
            k2 = _tenta_kps(e)
            if k2 is not None and kp is None:
                kp = k2
            elif pid is None and np.isscalar(e):
                pid = _int(e)
        if kp is not None:
            return (pid if pid is not None else pid_padrao, kp)
    return None


def _pessoas_de(val):
    """Lê todas as pessoas de um quadro."""
    if isinstance(val, dict):
        p = _pessoa(val)
        if p:
            return [p]
        out = []
        for k, v in val.items():
            p = _pessoa(v, pid_padrao=_int(k))
            if p:
                out.append(p)
        return out
    if isinstance(val, (list, tuple, np.ndarray)):
        p = _pessoa(val)
        if p:
            return [p]
        return [p for p in (_pessoa(e) for e in val) if p]
    return []


def extrair_registros(obj, ordem: str = "auto"):
    """Converte o conteúdo de um arquivo em lista de (quadro, pid, pontos)."""
    if isinstance(obj, dict):
        for c in ("annotations", "annotation", "data", "poses", "frames"):
            if c in obj and isinstance(obj[c], (dict, list)) and len(obj) <= 4:
                return extrair_registros(obj[c], ordem)

    regs = []
    if isinstance(obj, (list, tuple)):
        for it in obj:  # lista de registros com campo de quadro
            if isinstance(it, dict):
                low = {str(k).lower(): k for k in it}
                q = next((_int(it[low[c]]) for c in CHAVES_QUADRO if c in low), None)
                p = _pessoa(it)
                if q is not None and p:
                    regs.append((q, p[0], p[1]))
        if regs:
            return regs
        for q, it in enumerate(obj):  # lista indexada pelo quadro
            regs += [(q, pid, kp) for pid, kp in _pessoas_de(it)]
        return regs

    if not isinstance(obj, dict):
        raise ValueError(f"estrutura não reconhecida ({type(obj).__name__})")

    itens = [(_int(k), v) for k, v in obj.items()]
    itens = [(k, v) for k, v in itens if k is not None]
    if not itens:
        raise ValueError("dicionário sem chaves numéricas de quadro ou pessoa")

    def dois_niveis(v):
        return (isinstance(v, dict) and _pessoa(v) is None and
                any(_int(k2) is not None and _pessoa(v2) is not None
                    for k2, v2 in list(v.items())[:5]))

    if ordem == "auto":
        amostra = [v for _, v in itens[:20]]
        if all(dois_niveis(v) for v in amostra):
            internos = float(np.mean([len(v) for v in amostra]))
            tem_id = any(
                any(c in {str(x).lower() for x in v2} for c in CHAVES_ID)
                for v in amostra for v2 in list(v.values())[:3] if isinstance(v2, dict))
            ordem = "quadro-pessoa" if (tem_id or len(itens) > internos) else "pessoa-quadro"
        else:
            ordem = "quadro-pessoa"

    if ordem == "pessoa-quadro":
        for pid, v in itens:
            for kq, v2 in v.items():
                q, p = _int(kq), _pessoa(v2)
                if q is not None and p:
                    regs.append((q, pid, p[1]))
    else:
        for q, v in itens:
            regs += [(q, pid, kp) for pid, kp in _pessoas_de(v)]
    return regs


def ler_arquivo_pose(caminho: Path):
    caminho = Path(caminho)
    if caminho.suffix.lower() == ".json":
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    with open(caminho, "rb") as f:  # .pkl: só abra arquivos de fonte confiável
        return pickle.load(f)


def sanear(kp: np.ndarray, conf_nan: float = 1.0):
    """Trata valores indefinidos (NaN/inf) nos pontos.

    * x ou y indefinido -> ponto ausente (coordenadas e confiança zeradas);
    * só a confiança indefinida, com x e y válidos -> confiança = conf_nan.
      (No PoseLift, a confiança vem NaN em parte dos pontos, por exemplo nos
      quadros interpolados; com conf_nan=1 esses pontos são usados normalmente,
      com conf_nan=0 são descartados.)
    Devolve (pontos saneados, nº de confianças NaN, nº de x/y NaN).
    """
    kp = kp.astype(np.float32, copy=True)
    xy_ruim = ~np.isfinite(kp[..., :2]).all(-1)
    c_ruim = ~np.isfinite(kp[..., 2]) & ~xy_ruim
    kp[c_ruim, 2] = conf_nan
    kp[xy_ruim] = 0.0
    return kp, int(c_ruim.sum()), int(xy_ruim.sum())


def montar_video(nome: str, regs, rotulos=None, offset=None, conf_nan: float = 1.0):
    """Agrupa os registros por pessoa e alinha com os rótulos."""
    info = {"offset": 0, "descartados": 0, "sem_id": 0,
            "pontos": 0, "conf_nan": 0, "xy_nan": 0}
    if offset is None:
        offset = 0
        if regs and rotulos is not None:
            qs = [r[0] for r in regs]
            if min(qs) >= 1 and max(qs) == len(rotulos):
                offset = -1  # quadros numerados a partir de 1
    info["offset"] = offset

    por_quadro_sem_id: dict[int, int] = {}
    grupos: dict[int, dict[int, np.ndarray]] = {}
    for q, pid, kp in regs:
        if pid is None:
            info["sem_id"] += 1
            i = por_quadro_sem_id.get(q, 0)
            por_quadro_sem_id[q] = i + 1
            pid = -(i + 1)
        kp, n_c, n_xy = sanear(kp, conf_nan)
        info["pontos"] += kp.shape[0]
        info["conf_nan"] += n_c
        info["xy_nan"] += n_xy
        grupos.setdefault(pid, {}).setdefault(q + offset, kp)

    if rotulos is not None:
        n = len(rotulos)
    else:
        n = max((max(g) for g in grupos.values()), default=-1) + 1

    poses = {}
    for pid, g in grupos.items():
        qs = np.array(sorted(g), dtype=np.int64)
        ok = (qs >= 0) & (qs < n)
        info["descartados"] += int((~ok).sum())
        qs = qs[ok]
        if len(qs):
            poses[pid] = (qs, np.stack([g[q] for q in qs]).astype(np.float32))

    camera = nome.split("_")[0]
    return Video(nome, camera, poses, rotulos, n), info


# --------------------------------------------------------------------------- #
# Descoberta dos arquivos e divisão treino/teste
# --------------------------------------------------------------------------- #
def _divisao_de_pasta(caminho: Path, raiz: Path):
    partes = [p.lower() for p in caminho.relative_to(raiz).parts[:-1]]
    if any(p in ("train", "training", "treino") for p in partes):
        return "treino"
    if any(p in ("test", "testing", "teste") for p in partes):
        return "teste"
    return None


def _casar(stem: str, mapa: dict):
    if stem in mapa:
        return stem
    base = stem
    for suf in SUFIXOS_CONHECIDOS:
        if base.endswith(suf):
            base = base[: -len(suf)]
            if base in mapa:
                return base
    cands = [s for s in mapa
             if stem.startswith(s) and len(stem) > len(s) and not stem[len(s)].isdigit()]
    return max(cands, key=len) if cands else None


def _ler_divisao(caminho: Path) -> dict:
    """JSON {"treino": [...], "teste": [...]} ou CSV/TXT com linhas 'video,divisao'."""
    norm = {"train": "treino", "treino": "treino", "test": "teste", "teste": "teste"}
    mapa = {}
    caminho = Path(caminho)
    if caminho.suffix.lower() == ".json":
        with open(caminho, encoding="utf-8") as f:
            d = json.load(f)
        for k, nomes in d.items():
            for n in nomes:
                mapa[Path(str(n)).stem] = norm[k.lower()]
    else:
        with open(caminho, encoding="utf-8") as f:
            for linha in f:
                partes = [p.strip() for p in re.split(r"[,;\t ]+", linha.strip()) if p.strip()]
                if len(partes) >= 2 and partes[1].lower() in norm:
                    mapa[Path(partes[0]).stem] = norm[partes[1].lower()]
    return mapa


def localizar(raiz):
    """Lista arquivos de pose e de rótulo encontrados sob a pasta raiz."""
    raiz = Path(raiz)
    if not raiz.exists():
        raise FileNotFoundError(f"pasta não encontrada: {raiz}")
    poses = sorted(p for p in raiz.rglob("*") if p.is_file() and p.suffix.lower() in EXT_POSE)
    npys = sorted(p for p in raiz.rglob("*.npy") if p.is_file())
    rotulos = {}
    for p in npys:
        rotulos.setdefault(p.stem, p)
    divisoes = {p: _divisao_de_pasta(p, raiz) for p in poses}
    formato = "pastas" if any(divisoes.values()) else "arquivos"
    return {"raiz": raiz, "poses": poses, "rotulos": rotulos,
            "divisoes": divisoes, "formato": formato}


def carregar_conjunto(raiz, divisao=None, ordem="auto", offset=None, verbose=True,
                      somente=None, conf_nan=1.0):
    """Carrega todos os vídeos. Devolve {"treino": [...], "teste": [...]} e um relatório.

    somente: conjunto opcional de nomes de vídeo a carregar (acelera a avaliação).
    conf_nan: confiança atribuída a pontos com x/y válidos e confiança NaN.
    """
    info = localizar(raiz)
    mapa_div = _ler_divisao(divisao) if divisao else {}
    saida = {"treino": [], "teste": []}
    relatorio = {"formato": info["formato"], "ignorados": [], "offsets": {},
                 "sem_rotulo": [], "heuristica": False, "sem_id": 0, "descartados": 0,
                 "pontos": 0, "conf_nan": 0, "xy_nan": 0, "conf_nan_valor": conf_nan}

    for caminho in info["poses"]:
        stem = caminho.stem
        chave_rot = _casar(stem, info["rotulos"])
        nome = chave_rot or stem
        if somente is not None and nome not in somente:
            continue
        try:
            regs = extrair_registros(ler_arquivo_pose(caminho), ordem)
        except Exception as e:  # noqa: BLE001 - arquivo que não é de pose
            relatorio["ignorados"].append(f"{caminho.name}: {e}")
            continue
        if not regs:
            relatorio["ignorados"].append(f"{caminho.name}: nenhuma pose reconhecida")
            continue
        rot = None
        if chave_rot is not None:
            rot = np.load(info["rotulos"][chave_rot]).astype(np.int64).ravel()
        else:
            relatorio["sem_rotulo"].append(nome)
        video, inf = montar_video(nome, regs, rot, offset, conf_nan)
        relatorio["offsets"][nome] = inf["offset"]
        for k in ("sem_id", "descartados", "pontos", "conf_nan", "xy_nan"):
            relatorio[k] += inf[k]

        div = mapa_div.get(nome) or mapa_div.get(stem) or info["divisoes"][caminho]
        if div is None:
            relatorio["heuristica"] = True
            div = "teste" if (rot is not None and rot.any()) else "treino"
        saida[div].append(video)

    if verbose:
        _avisos(relatorio)
    return saida, relatorio


def _avisos(r):
    if r["ignorados"]:
        print(f"[aviso] {len(r['ignorados'])} arquivo(s) ignorado(s), ex.: {r['ignorados'][0]}")
    if r["heuristica"]:
        print("[aviso] Divisão treino/teste inferida pela regra 'vídeo com furto -> teste'. "
              "Se houver um arquivo de divisão oficial, use --divisao.")
    if r["sem_id"]:
        print(f"[aviso] {r['sem_id']} poses sem ID de pessoa; o ID foi atribuído pela ordem no "
              "quadro, o que quebra o rastreamento. Confira com inspecionar.py.")
    if any(v != 0 for v in r["offsets"].values()):
        print("[info] Quadros numerados a partir de 1 detectados; deslocados em -1 para "
              "alinhar com os rótulos.")
    if r["descartados"]:
        print(f"[aviso] {r['descartados']} poses fora do intervalo dos rótulos foram descartadas.")
    if r["pontos"] and (r["conf_nan"] or r["xy_nan"]):
        print(f"[info] Valores indefinidos (NaN): confiança em {100 * r['conf_nan'] / r['pontos']:.1f}% "
              f"dos pontos (usada confiança = {r['conf_nan_valor']}); x/y em "
              f"{100 * r['xy_nan'] / r['pontos']:.1f}% (tratados como ausentes).")


# --------------------------------------------------------------------------- #
# Trajetórias, normalização e janelas
# --------------------------------------------------------------------------- #
def _interpolar(q, k, alvo):
    res = np.empty((len(alvo), V * 3), np.float32)
    plano = k.reshape(len(q), -1)
    for c in range(plano.shape[1]):
        res[:, c] = np.interp(alvo, q, plano[:, c])
    return res.reshape(len(alvo), V, 3)


def trajetorias(video: Video, cfg: Config):
    """Sequências contínuas por pessoa: lista de (pid, quadros, pontos (n,17,3))."""
    out = []
    for pid, (q, kp) in video.poses.items():
        cortes = np.where(np.diff(q) > cfg.lacuna_max + 1)[0] + 1
        for qs, ks in zip(np.split(q, cortes), np.split(kp, cortes)):
            if qs[-1] - qs[0] + 1 < cfg.T:
                continue
            completo = np.arange(qs[0], qs[-1] + 1)
            if len(completo) != len(qs):
                ks = _interpolar(qs, ks, completo)
            out.append((pid, completo, ks))
    return out


def remover_pontos(trajs, fracao: float, rng: np.random.Generator):
    """Oclusão simulada: zera a confiança de uma fração das articulações."""
    out = []
    for pid, q, k in trajs:
        k = k.copy()
        k[..., 2][rng.random(k.shape[:2]) < fracao] = 0.0
        out.append((pid, q, k))
    return out


def normalizar(j: np.ndarray, cfg: Config) -> np.ndarray:
    """Centraliza no quadril e divide pelo tronco. j: (..., T, 17, 3)."""
    j = j.astype(np.float32, copy=True)
    xy, c = j[..., :2], j[..., 2]
    val = c >= cfg.conf_min

    def media(idx):
        return xy[..., idx, :].mean(-2), val[..., idx].all(-1)

    quadril, ok_q = media(QUADRIS)
    ombro, ok_o = media(OMBROS)
    w = val[..., None].astype(np.float32)
    media_val = (xy * w).sum(-2) / np.maximum(w.sum(-2), 1e-6)
    centro = np.where(ok_q[..., None], quadril, media_val)

    tronco = np.linalg.norm(ombro - quadril, axis=-1)
    ok_t = ok_q & ok_o
    n = ok_t.sum(-1)
    escala = np.where(n > 0, (tronco * ok_t).sum(-1) / np.maximum(n, 1), np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        ys = np.where(val, xy[..., 1], np.nan)
        alt = (np.nanmax(ys, axis=(-1, -2)) - np.nanmin(ys, axis=(-1, -2))) / 3.0
    escala = np.where(np.isfinite(escala) & (escala > 1e-3), escala, alt)
    escala = np.where(np.isfinite(escala) & (escala > 1e-3), escala, 1.0)

    if cfg.centro == "janela":
        ok_c = ok_q | val.any(-1)
        cj = (centro * ok_c[..., None]).sum(-2) / np.maximum(ok_c.sum(-1), 1)[..., None]
        centro = np.broadcast_to(cj[..., None, :], centro.shape)

    xy = (xy - centro[..., None, :]) / escala[..., None, None, None]
    xy[~val] = 0.0
    j[..., :2] = xy
    j[..., 2] = np.where(val, c, 0.0)
    return j


def gerar_janelas(trajs, cfg: Config, passo: int, video_idx: int = 0):
    """Janelas normalizadas (N, 3, T, 17) e metadados (N, 3) = [vídeo, pid, quadro inicial]."""
    blocos, meta = [], []
    for pid, q, k in trajs:
        n = len(q)
        inicios = list(range(0, n - cfg.T + 1, passo))
        if inicios[-1] != n - cfg.T:
            inicios.append(n - cfg.T)
        for s in inicios:
            blocos.append(k[s:s + cfg.T])
            meta.append((video_idx, pid, int(q[s])))
    if not blocos:
        return np.zeros((0, 3, cfg.T, V), np.float32), np.zeros((0, 3), np.int64)
    x = normalizar(np.stack(blocos), cfg).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x), np.asarray(meta, np.int64)


def janelas_de_videos(videos, cfg: Config, passo: int, excluir_anomalias=True):
    """Janelas de vários vídeos. Remove janelas que tocam quadros rotulados como furto."""
    xs, ms = [], []
    removidas = 0
    for i, v in enumerate(videos):
        x, m = gerar_janelas(trajetorias(v, cfg), cfg, passo, i)
        if excluir_anomalias and v.rotulos is not None and v.rotulos.any() and len(x):
            acum = np.concatenate([[0], np.cumsum(v.rotulos)])
            fim = np.minimum(m[:, 2] + cfg.T, len(v.rotulos))
            limpa = (acum[fim] - acum[m[:, 2]]) == 0
            removidas += int((~limpa).sum())
            x, m = x[limpa], m[limpa]
        xs.append(x)
        ms.append(m)
    if removidas:
        print(f"[info] {removidas} janelas de treino com quadros de furto foram removidas.")
    return np.concatenate(xs), np.concatenate(ms)
