import numpy as np

from tcc.data.schema import VideoPoses
from tcc.rules.zones import (
    ExitMonitor,
    ExitRule,
    Visit,
    foot_points,
    in_polygon,
    runs,
    track_visits,
    unpaid_exits,
)

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
RULE = ExitRule(min_shelf_frames=30, min_checkout_frames=15, payment_grace_frames=60)


def test_in_polygon_square_and_concave() -> None:
    pts = np.array([[5.0, 5.0], [15.0, 5.0], [-1.0, 5.0]])
    assert in_polygon(pts, SQUARE).tolist() == [True, False, False]
    ell = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert in_polygon(np.array([[2.0, 8.0], [8.0, 8.0]]), ell).tolist() == [True, False]


def test_foot_points_use_ankles_only_when_both_are_confident() -> None:
    bbox = np.array([[0, 0, 10, 20], [0, 0, 10, 20]], np.float32)
    kp = np.zeros((2, 17, 3), np.float32)
    kp[:, 15] = [2, 18, 0.9]
    kp[:, 16] = [4, 18, 0.9]
    kp[1, 16, 2] = 0.1
    assert foot_points(bbox, kp, 0.3).tolist() == [[3.0, 18.0], [5.0, 20.0]]


def test_runs_join_short_gaps_and_drop_short_runs() -> None:
    frames = np.array([0, 1, 2, 5, 6, 20, 21])
    assert runs(frames, max_gap=2, min_frames=3) == [(0, 6)]
    assert runs(frames, max_gap=1, min_frames=2) == [(0, 2), (5, 6), (20, 21)]


def _video(path: list[tuple[float, float]]) -> VideoPoses:
    n = len(path)
    bbox = np.array([[x - 5, y - 40, x + 5, y] for x, y in path], np.float32)
    return VideoPoses(
        video_id="v",
        camera="c",
        n_frames=n,
        frame_idx=np.arange(n, dtype=np.int64),
        track_id=np.zeros(n, np.int64),
        bbox=bbox,
        keypoints=np.zeros((n, 17, 3), np.float32),
        frame_labels=None,
    )


def test_track_visits_follow_the_feet_through_the_zones() -> None:
    zones = {"shelf": SQUARE, "exit": [(20, 0), (30, 0), (30, 10), (20, 10)]}
    path = [(5.0, 5.0)] * 40 + [(15.0, 5.0)] * 10 + [(25.0, 5.0)] * 20
    visits = track_visits(_video(path), zones, 0.3, max_gap=2, min_frames=5)
    assert visits == [Visit(0, "shelf", 0, 39), Visit(0, "exit", 50, 69)]


def _visits(*rows: tuple[str, int, int]) -> list[Visit]:
    return [Visit(0, z, a, b) for z, a, b in rows]


def test_shelf_then_exit_without_checkout_alerts_once() -> None:
    v = _visits(("shelf", 0, 59), ("exit", 100, 120))
    assert [(a.frame, a.shelf_frames, a.first_shelf) for a in unpaid_exits(v, [], RULE)] == [
        (100, 60, 0)
    ]


def test_checkout_or_a_payment_or_a_short_shelf_visit_clears_the_exit() -> None:
    assert (
        unpaid_exits(_visits(("shelf", 0, 59), ("checkout", 70, 90), ("exit", 100, 110)), [], RULE)
        == []
    )
    assert unpaid_exits(_visits(("shelf", 0, 59), ("exit", 100, 110)), [150], RULE) == []
    assert unpaid_exits(_visits(("shelf", 0, 59), ("exit", 100, 110)), [161], RULE) != []
    assert unpaid_exits(_visits(("shelf", 0, 9), ("exit", 100, 110)), [], RULE) == []


def test_online_monitor_equals_the_offline_rule() -> None:
    zones = {
        "shelf": SQUARE,
        "checkout": [(40, 0), (50, 0), (50, 10), (40, 10)],
        "exit": [(20, 0), (30, 0), (30, 10), (20, 10)],
    }
    # unpaid trip, then a paid trip (checkout), then an unpaid trip cancelled by a sale
    path = (
        [(5.0, 5.0)] * 50 + [(25.0, 5.0)] * 20 + [(60.0, 5.0)] * 100
        + [(5.0, 5.0)] * 50 + [(45.0, 5.0)] * 30 + [(25.0, 5.0)] * 20 + [(60.0, 5.0)] * 100
        + [(5.0, 5.0)] * 50 + [(25.0, 5.0)] * 20 + [(60.0, 5.0)] * 30
    )  # fmt: skip
    video = _video(path)
    payments = [len(path) - 40]
    offline = unpaid_exits(track_visits(video, zones, 0.3, 2, 5), payments, RULE)
    monitor = ExitMonitor(zones, 0.3, 2, 5, RULE)
    online = []
    for q in range(video.n_frames):
        if q in payments:
            monitor.pay(q)
        online += monitor.push(q, video.track_id[[q]], video.bbox[[q]], video.keypoints[[q]])
    online += monitor.flush()
    assert [a.frame for a in offline] == [50]
    assert online == offline


def test_each_trip_is_judged_on_its_own() -> None:
    v = _visits(
        ("shelf", 0, 59), ("checkout", 60, 80), ("exit", 90, 95),
        ("shelf", 200, 260), ("exit", 300, 310),
    )  # fmt: skip
    assert [a.frame for a in unpaid_exits(v, [], RULE)] == [300]
