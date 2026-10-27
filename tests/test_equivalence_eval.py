"""Port equivalence of the evaluation (docs/tcc/PORTE_POC_LUIZ.md, criterion "Pontuação e
métricas"; session 3, task B5), on the validation scores of the original reference run.

The frame scores are compared with the arrays the original wrote (`val_ref.npz`). The
threshold, alert and metric functions of the original `avaliar.py` run in a subprocess
(no import, as in test_synthetic_fixture) on those validation scores; label-based
metrics use synthetic labels, because validation has no theft. Options that reproduce
the original: `max_cover`, fill with the validation minimum, one sigma of 5, alerts with
no minimum duration and every frame scored. The original counts `score >= threshold`
and we count `score > threshold` (CLAUDE.md "escore > limiar"); thresholds that equal a
score exactly are reported and left out of the comparison.
Needs data/reference/poc_luiz_seed0 (not versioned); skipped without it.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tcc.eval.alerts import alerts
from tcc.eval.events import event_metrics
from tcc.eval.frame_scores import post_process, window_frame_scores
from tcc.eval.metrics import auc_pr, auc_roc, eer, threshold_metrics
from tcc.eval.threshold import (
    alerts_per_hour,
    candidate_thresholds,
    quantile_threshold,
    threshold_for_rate,
)

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference" / "poc_luiz_seed0"
LEGACY = ROOT / "legacy" / "poc_luiz"
T, FPS, MERGE, TOLERANCE = 24, 15, 30, 15  # original defaults: --juntar-s 2, --tolerancia-s 1
QUANTILE = 0.95  # original default of --quantil
RATES = [1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
PT_CONFUSION = {"tp": "VP", "fp": "FP", "fn": "FN", "tn": "VN"}

RUNNER = """
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from avaliar import alertas, alertas_por_hora, avaliar_eventos, eer, limiar_da_validacao
from avaliar import metricas_limiar
from sklearn.metrics import average_precision_score, roc_auc_score
z = np.load(sys.argv[2])
n = int(z["n"])
s = [z[f"s{i}"] for i in range(n)]
y = [z[f"y{i}"] for i in range(n)]
sample = np.concatenate(s)
yy = np.concatenate(y)
out = {
    "quantile": limiar_da_validacao(s, 30, 15, quantil=float(z["q"]), amostra=sample)[0],
    "rate": {str(r): limiar_da_validacao(s, 30, 15, alvo_hora=float(r), amostra=sample)[0]
             for r in z["rates"]},
    "per_hour": [alertas_por_hora(s, float(t), 30, 15) for t in z["thresholds"]],
    "alerts": [[[list(map(int, a)) for a in alertas(v, float(t), 30)] for v in s]
               for t in z["thresholds"]],
    "eer": eer(yy, sample),
    "auc_roc": float(roc_auc_score(yy, sample)),
    "average_precision": float(average_precision_score(yy, sample)),
    "at_threshold": [metricas_limiar(yy, sample, float(t)) for t in z["thresholds"]],
    "events": [avaliar_eventos(s, y, float(t), 30, 15) for t in z["thresholds"]],
}
json.dump(out, open(sys.argv[3], "w"))
"""

pytestmark = pytest.mark.skipif(
    not (REF / "val_ref.npz").exists() or not (LEGACY / "avaliar.py").exists(),
    reason="reference run or legacy copy not available",
)


def _reference() -> tuple[list[np.ndarray], list[np.ndarray], np.lib.npyio.NpzFile]:
    ref = np.load(REF / "val_ref.npz")
    vids = [str(v) for v in ref["videos"]]
    raw = [ref[f"{v}__frame_scores_raw"] for v in vids]
    post = [ref[f"{v}__frame_scores_post"] for v in vids]
    return raw, post, ref


def test_max_cover_frame_scores_reproduce_the_original_on_validation() -> None:
    raw, _, ref = _reference()
    for vid, expected in zip(ref["videos"], raw, strict=True):
        ours = window_frame_scores(
            ref[f"{vid}__window_meta"], ref[f"{vid}__window_scores"], T, len(expected),
            "max_cover",
        )  # fmt: skip
        np.testing.assert_array_equal(ours, expected)


def test_post_processing_reproduces_the_original_on_validation() -> None:
    raw, post, ref = _reference()
    ours = post_process(raw, float(ref["piso"]), [float(ref["sigma"])])
    for a, b in zip(ours, post, strict=True):
        np.testing.assert_array_equal(a, b)


def _synthetic_labels(post: list[np.ndarray]) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    out = []
    for s in post:
        y = np.zeros(len(s), np.int64)
        a = int(rng.integers(0, max(1, len(s) - 90)))
        y[a : a + 90] = 1
        out.append(y)
    return out


def test_threshold_alerts_and_metrics_reproduce_the_original_on_validation(
    tmp_path: Path,
) -> None:
    _, post, _ = _reference()
    labels = _synthetic_labels(post)
    sample, y = np.concatenate(post), np.concatenate(labels)
    thresholds = candidate_thresholds(sample, 0.5, 0.99999, 400)[::20]
    inputs, output = tmp_path / "inputs.npz", tmp_path / "output.json"
    arrays = {f"s{i}": s for i, s in enumerate(post)} | {f"y{i}": v for i, v in enumerate(labels)}
    np.savez(inputs, n=len(post), q=QUANTILE, rates=RATES, thresholds=thresholds, **arrays)
    subprocess.run(
        [sys.executable, "-c", RUNNER, str(LEGACY), str(inputs), str(output)],
        cwd=LEGACY,
        check=True,
        capture_output=True,
    )
    theirs = json.loads(output.read_text())

    every = [(s, np.ones(len(s), np.bool_)) for s in post]
    assert quantile_threshold(sample, QUANTILE) == theirs["quantile"]
    candidates = candidate_thresholds(sample, 0.5, 0.99999, 400)
    for r in RATES:
        assert threshold_for_rate(every, r, candidates, 1, MERGE, FPS)[0] == theirs["rate"][str(r)]

    tied = [bool((sample == t).any()) for t in thresholds]
    assert sum(tied) < len(thresholds) // 2, "too many exact ties to compare"
    for i, t in enumerate(thresholds):
        if tied[i]:
            continue
        assert alerts_per_hour(every, float(t), 1, MERGE, FPS) == theirs["per_hour"][i]
        ours = [[[a.start, a.end] for a in alerts(s, m, float(t), 1, MERGE)] for s, m in every]
        assert ours == theirs["alerts"][i]
        m = threshold_metrics(y, sample, float(t))
        o = theirs["at_threshold"][i]
        confusion = o["matriz_confusao"]
        assert m["confusion"] == {k: confusion[v] for k, v in PT_CONFUSION.items()}
        for ours_key, their_key in [
            ("precision", "precisao"), ("recall", "revocacao"), ("specificity", "especificidade"),
            ("f1", "f1"), ("h_prs", "h_prs"),
        ]:  # fmt: skip
            assert m[ours_key] == pytest.approx(o[their_key], rel=1e-12, abs=1e-15)
        intervals = [
            [(a.start, a.end) for a in alerts(s, mk, float(t), 1, MERGE)] for s, mk in every
        ]
        e, oe = event_metrics(labels, intervals, TOLERANCE), theirs["events"][i]
        assert (e["events"], e["events_detected"], e["alerts"], e["false_alerts"]) == (
            oe["eventos"], oe["eventos_detectados"], oe["alertas"], oe["alertas_falsos"],
        )  # fmt: skip

    assert eer(y, sample) == theirs["eer"]
    assert auc_roc(y, sample) == theirs["auc_roc"]
    assert auc_pr(y, sample, "theft", "average_precision") == theirs["average_precision"]
