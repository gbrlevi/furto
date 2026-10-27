"""Characterization against the STG-NF scoring code distributed with PoseLift (ADR-0008).

Runs `STG-NF/utils/scoring_utils.py` from the authors' Drive (not versioned; CC BY-NC
4.0) in a subprocess, unmodified, on synthetic inputs, and compares it with our frame
protocol: score at the window centre, maximum over people, frames without a score at the
most normal score of the whole set, gaussian with sigma 1 to 6 in sequence. STG-NF scores
normality and labels normal as 1, so it receives the negated anomaly score.
`tqdm` and the `dataset` module (only `shanghaitech_hr_skip`) are stubbed: they are
not needed by the scoring and pull in the training stack.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tcc.eval.frame_scores import most_normal, post_process, window_frame_scores
from tcc.eval.metrics import auc_pr, auc_roc, eer

ROOT = Path(__file__).resolve().parents[1]
STGNF = ROOT / "data" / "raw" / "poselift" / "STG-NF"
T = 24
SIGMAS = [1, 2, 3, 4, 5, 6]
# Declared tolerance of the EER comparison: STG-NF reports the false positive rate at the
# closest ROC point, we report the mean of both rates there; with a few hundred frames per
# class the ROC grid step is below 0.01.
EER_TOLERANCE = 0.01

RUNNER = """
import importlib.util, sys, types
import numpy as np
stgnf, inputs, output = sys.argv[1:4]
tqdm = types.ModuleType("tqdm"); tqdm.tqdm = lambda x, **k: x
dataset = types.ModuleType("dataset"); dataset.shanghaitech_hr_skip = lambda *a: False
sys.modules["tqdm"], sys.modules["dataset"] = tqdm, dataset
spec = importlib.util.spec_from_file_location("scoring_utils", stgnf + "/utils/scoring_utils.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
z = np.load(inputs)
args = types.SimpleNamespace(dataset="PoseLift", seg_len=int(z["seg_len"]))
auc_roc, scores, auc_pr, eer, _ = m.score_dataset(
    z["normality"], z["metadata"], args=args, save_results=False)
np.savez(output, auc_roc=auc_roc, scores=scores, auc_pr=auc_pr, eer=eer)
"""

pytestmark = pytest.mark.skipif(
    not (STGNF / "utils" / "scoring_utils.py").exists(), reason="STG-NF code not available"
)


def _synthetic(rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """Clips with 0 to 3 people, stride-1 windows, theft intervals with higher scores and
    frames with nobody; one clip has no window at all."""
    clips = []
    for c, n_frames in enumerate([180, 240, 150, 200, 120]):
        labels = np.zeros(n_frames, np.int64)
        if c % 2 == 0:
            a = int(rng.integers(30, n_frames - 60))
            labels[a : a + 40] = 1
        rows, scores = [], []
        for person in range(0 if c == 4 else int(rng.integers(1, 4))):
            first = int(rng.integers(0, 40))
            length = int(rng.integers(T + 10, n_frames - first - T // 2))
            for start in range(first, first + length - T + 1):
                rows.append((c, person, start))
                centre = start + T // 2
                scores.append(rng.gamma(2.0, 0.02) + 0.05 * labels[centre] * rng.random())
        clips.append(
            {
                "name": f"1_{101 + c}",
                "labels": labels,
                "meta": np.asarray(rows, np.int64).reshape(-1, 3),
                "scores": np.asarray(scores, np.float64),
            }
        )
    return clips


def test_our_protocol_reproduces_the_distributed_stgnf_scoring(tmp_path: Path) -> None:
    clips = _synthetic(np.random.default_rng(11))
    gt = tmp_path / "data" / "PoseLift" / "gt" / "test_frame_mask"
    gt.mkdir(parents=True)
    metadata, normality = [], []
    for clip in clips:
        np.save(gt / f"{clip['name']}.npy", clip["labels"])
        scene, number = (int(v) for v in clip["name"].split("_"))
        for (_, person, start), s in zip(clip["meta"], clip["scores"], strict=True):
            metadata.append((scene, number, person, start))
            normality.append(-s)
    inputs, output = tmp_path / "inputs.npz", tmp_path / "output.npz"
    np.savez(
        inputs,
        metadata=np.asarray(metadata, np.int64),
        normality=np.asarray(normality),
        seg_len=T,
    )
    subprocess.run(
        [sys.executable, "-c", RUNNER, str(STGNF), str(inputs), str(output)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    theirs = np.load(output)

    clips.sort(key=lambda c: f"{c['name']}.npy")  # STG-NF reads the label files sorted
    raw = [
        window_frame_scores(c["meta"], c["scores"], T, len(c["labels"]), "center") for c in clips
    ]
    ours = post_process(raw, most_normal(raw), SIGMAS)
    y, s = np.concatenate([c["labels"] for c in clips]), np.concatenate(ours)

    np.testing.assert_allclose(-s, theirs["scores"], rtol=1e-12, atol=1e-15)
    assert auc_roc(y, s) == pytest.approx(float(theirs["auc_roc"]), abs=1e-12)
    assert auc_pr(y, s, "normal", "trapezoid") == pytest.approx(float(theirs["auc_pr"]), abs=1e-12)
    assert abs(eer(y, s) - float(theirs["eer"])) < EER_TOLERANCE
