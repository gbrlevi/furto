from pathlib import Path

import numpy as np
import pytest

from tcc.data.merl import load_labels, merge, split_of, to_fps
from tcc.eval.intervals import iou, match

LABEL = Path("data/raw/merl/labels/Labels_MERL_Shopping_Dataset/1_1_label.mat")


@pytest.mark.skipif(not LABEL.exists(), reason="MERL labels not downloaded")
def test_load_labels_reads_the_five_actions_zero_based() -> None:
    labels = load_labels(LABEL)
    assert list(labels) == ["reach", "retract", "hand_in_shelf", "inspect_product", "inspect_shelf"]
    assert [len(v) for v in labels.values()] == [22, 22, 3, 12, 6]
    assert labels["reach"][0].tolist() == [155, 182]


def test_split_follows_the_readme() -> None:
    assert [split_of(s) for s in (1, 20, 21, 26, 27, 41)] == [
        "train",
        "train",
        "val",
        "val",
        "test",
        "test",
    ]


def test_to_fps_and_merge() -> None:
    assert to_fps(np.array([[155, 182]]), 15.0).tolist() == [[77, 91]]
    spans = np.array([[10, 20], [22, 30], [50, 60], [55, 58]])
    assert merge(spans, max_gap=1) == [(10, 30), (50, 60)]
    assert merge(spans, max_gap=0) == [(10, 20), (22, 30), (50, 60)]


def test_iou_and_greedy_match() -> None:
    assert iou((0, 9), (5, 14)) == 5 / 15
    assert iou((0, 4), (5, 9)) == 0.0
    pred = [(0, 9), (100, 109), (200, 201)]
    truth = [(1, 10), (5, 14), (300, 310)]
    assert match(pred, truth, 0.3) == (1, 2, 2)
    assert match([], truth, 0.3) == (0, 0, 3)
