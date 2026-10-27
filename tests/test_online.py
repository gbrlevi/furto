"""Online scoring equals the offline scoring of the same frames (session 3B, task 6)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tcc.data.schema import VideoPoses
from tcc.eval.alerts import alerts
from tcc.model.stgcn_ae import STGCNAutoencoder
from tcc.model.train import window_scores
from tcc.pipeline.online import AlertLog, OnlineScorer
from tcc.pipeline.scoring import (
    NOT_SCORABLE,
    SCORED,
    frame_scores,
    last_frame_reasons,
    last_frame_scores,
)
from tcc.preprocess.quality import SCORABLE
from tcc.preprocess.windows import treated_windows
from tcc.utils.config import load_config
from tests.fixtures.synthetic import person

ROOT = Path(__file__).resolve().parents[1]
SIZE = (1920, 1080)
RTOL = 1e-6  # declared: CPU float32, batches of different size online and offline


def _video() -> VideoPoses:
    """Three people with gaps of 3, k and k + 1 frames, a low-confidence stretch (filtered
    by the quality gate) and one person too short for a window."""
    rng = np.random.default_rng(4)
    n = 260
    rows_q, rows_t, rows_k = [], [], []
    spans = {1: (0, 220), 2: (30, 260), 3: (100, 118)}
    for tid, (a, b) in spans.items():
        kp = person(b - a, rng)
        frames = np.arange(a, b)
        keep = np.ones(b - a, np.bool_)
        if tid == 1:
            keep[40:43] = False  # gap of 3
            keep[90:98] = False  # gap of k = 8
            keep[150:159] = False  # gap of k + 1: a new piece
        if tid == 2:
            kp[190:230, :, 2] = 0.12  # frames 220-259, alone: below the gate threshold
        rows_q.append(frames[keep])
        rows_t.append(np.full(keep.sum(), tid))
        rows_k.append(kp[keep])
    q, t, k = np.concatenate(rows_q), np.concatenate(rows_t), np.concatenate(rows_k)
    xy = k[..., :2]
    boxes = np.concatenate([xy.min(1) - 5, xy.max(1) + 5], 1).astype(np.float32)
    order = np.lexsort((t, q))
    return VideoPoses("synthetic", "c", n, q[order], t[order], boxes[order], k[order], None)


@pytest.fixture(scope="module")
def setup() -> tuple[VideoPoses, dict, STGCNAutoencoder]:
    torch.manual_seed(0)
    model = STGCNAutoencoder().eval()
    treatment = load_config(ROOT / "configs" / "c1_base.yaml")["treatment"]
    return _video(), treatment, model


def _offline(video: VideoPoses, t: dict, model: STGCNAutoencoder) -> tuple:
    w = treated_windows(video, t, 1, SIZE)
    ok = w.reason == SCORABLE
    scores = window_scores(model, w.x[ok], torch.device("cpu"), "conf", 1024)
    per_person = last_frame_scores(w.meta[ok], scores, t["T"])
    per_reason = last_frame_reasons(w.meta, w.reason, t["T"])
    return per_person, per_reason, frame_scores(per_person, per_reason, video.n_frames)


def _online(video: VideoPoses, t: dict, model: STGCNAutoencoder) -> tuple:
    scorer = OnlineScorer(model, torch.device("cpu"), t, SIZE, "conf", 1024)
    finals = []
    for q in range(video.n_frames):
        rows = video.frame_idx == q
        finals += scorer.push(q, video.track_id[rows], video.bbox[rows], video.keypoints[rows])
    finals += scorer.flush()
    return scorer, finals


def test_online_frame_scores_states_and_reasons_equal_offline(setup: tuple) -> None:
    video, t, model = setup
    per_person, per_reason, fs = _offline(video, t, model)
    scorer, finals = _online(video, t, model)
    assert [f.frame for f in finals] == list(range(video.n_frames))
    assert scorer.per_reason == per_reason
    assert scorer.per_person.keys() == per_person.keys()
    for key, s in per_person.items():
        assert scorer.per_person[key] == pytest.approx(s, rel=RTOL)
    state = np.array([f.state for f in finals])
    np.testing.assert_array_equal(state, fs.state)
    np.testing.assert_array_equal(np.array([f.reason for f in finals]), fs.reason)
    score = np.array([f.score for f in finals])
    np.testing.assert_allclose(score[state == SCORED], fs.score[fs.state == SCORED], rtol=RTOL)
    # the fixture exercises every case
    assert (fs.state == NOT_SCORABLE).any() and (fs.state == SCORED).any()
    assert not any(tid == 3 for tid, _ in per_person)  # id 3: shorter than T


def test_online_alerts_equal_offline(setup: tuple) -> None:
    video, t, model = setup
    _, _, fs = _offline(video, t, model)
    scored = fs.state == SCORED
    threshold = float(np.quantile(fs.score[scored], 0.4))
    expected = alerts(fs.score, scored, threshold, 15, 30)
    assert expected, "the fixture must raise at least one alert"
    _, finals = _online(video, t, model)
    log = AlertLog(threshold, 15, 30)
    events = [e for f in finals for e in log.add(f)]
    assert log.alerts() == expected
    fired = [e for e in events if e["event"] == "alert_fired"]
    assert [e["fired"] for e in fired] == [a.fired for a in expected]
    assert all(e["at_frame"] == e["fired"] for e in fired)


def test_ids_gone_for_more_than_k_frames_are_dropped(setup: tuple) -> None:
    video, t, model = setup
    scorer = OnlineScorer(model, torch.device("cpu"), t, SIZE, "conf", 1024)
    for q in range(video.n_frames):
        rows = video.frame_idx == q
        scorer.push(q, video.track_id[rows], video.bbox[rows], video.keypoints[rows])
        if q == 230:  # id 1 last seen at 219
            assert 1 not in scorer.active_ids() and 2 in scorer.active_ids()
