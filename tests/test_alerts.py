"""Alert logic (session 2D, task 6): minimum duration and merging, on synthetic sequences."""

import numpy as np

from tcc.eval.alerts import Alert, alerts

THR = 1.0


def _sequence(*runs: tuple[int, int], n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    score = np.zeros(n)
    for start, end in runs:
        score[start:end] = 2.0
    return score, np.ones(n, np.bool_)


def test_fourteen_frames_above_do_not_alert() -> None:
    assert alerts(*_sequence((10, 24)), THR, 15, 30) == []


def test_fifteen_frames_above_alert_on_the_fifteenth() -> None:
    assert alerts(*_sequence((10, 25)), THR, 15, 30) == [Alert(10, 25, 24)]


def test_alerts_separated_by_29_frames_are_merged() -> None:
    assert alerts(*_sequence((10, 25), (54, 70)), THR, 15, 30) == [Alert(10, 70, 24)]


def test_alerts_separated_by_30_frames_stay_apart() -> None:
    out = alerts(*_sequence((10, 25), (55, 70)), THR, 15, 30)
    assert out == [Alert(10, 25, 24), Alert(55, 70, 69)]


def test_not_scored_frames_break_the_count() -> None:
    score, scored = _sequence((10, 30))
    scored[20] = False  # e.g. only filtered windows in this frame
    assert alerts(score, scored, THR, 15, 30) == []
    assert alerts(score, np.ones_like(scored), THR, 15, 30) == [Alert(10, 30, 24)]


def test_score_equal_to_threshold_is_not_above() -> None:
    score, scored = _sequence()
    score[10:40] = THR
    assert alerts(score, scored, THR, 15, 30) == []
