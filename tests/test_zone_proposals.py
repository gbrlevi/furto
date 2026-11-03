import numpy as np

from tcc.rules.zone_proposals import box_iou, nms, propose

KINDS = ["shelf", "shelf", "checkout"]


def test_box_iou_of_identical_disjoint_and_half_overlapping_boxes() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [5.0, 0.0, 15.0, 10.0]])
    assert np.allclose(box_iou(a, b), [[1.0, 0.0, 50 / 150]])


def test_nms_drops_the_weaker_of_two_overlapping_boxes_only() -> None:
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    assert nms(boxes, np.array([0.2, 0.3, 0.1]), 0.5) == [1, 2]


def test_propose_clips_drops_huge_boxes_and_keeps_the_best_per_kind() -> None:
    boxes = np.array(
        [
            [-10.0, 0.0, 50.0, 40.0],  # shelf, clipped to x1 = 0
            [0.0, 0.0, 100.0, 100.0],  # shelf over the whole image: dropped
            [60.0, 60.0, 90.0, 90.0],  # shelf, weaker than the first
            [70.0, 0.0, 90.0, 20.0],  # checkout
            [71.0, 0.0, 91.0, 20.0],  # checkout, weaker and over the first
        ]
    )
    scores = np.array([0.3, 0.9, 0.2, 0.15, 0.12])
    labels = np.array([0, 1, 0, 2, 2])
    out = propose(boxes, scores, labels, KINDS, 100, 100, 0.3, 0.5, {"shelf": 1, "checkout": 2})
    assert out["shelf"] == [[[0.0, 0.0], [50.0, 0.0], [50.0, 40.0], [0.0, 40.0]]]
    assert out["checkout"] == [[[70.0, 0.0], [90.0, 0.0], [90.0, 20.0], [70.0, 20.0]]]


def test_propose_returns_empty_kinds_without_boxes() -> None:
    empty = np.zeros((0, 4))
    out = propose(empty, np.zeros(0), np.zeros(0, np.int64), KINDS, 10, 10, 0.3, 0.5,
                  {"shelf": 3, "checkout": 1})  # fmt: skip
    assert out == {"shelf": [], "checkout": []}
