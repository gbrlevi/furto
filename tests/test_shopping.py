import numpy as np

from tcc.data.schema import VideoPoses
from tcc.rules.shopping import (
    Contact,
    Release,
    ShopAlert,
    Take,
    after_contact,
    contacts,
    hand_conf,
    hand_points,
    hands_in_zone,
    near_body,
    releases,
    shop_alerts,
    takes,
)

SHELF = [(100.0, 0.0), (200.0, 0.0), (200.0, 100.0), (100.0, 100.0)]
BASKET = [(0.0, 150.0), (50.0, 150.0), (50.0, 200.0), (0.0, 200.0)]


def _person(n: int, track: int = 0) -> VideoPoses:
    """One standing person at x = 50: shoulders at y = 40, hips at y = 100, hands at the hips."""
    kp = np.zeros((n, 17, 3), np.float32)
    kp[:, :, 2] = 0.9
    kp[:, 5, :2], kp[:, 6, :2] = (40, 40), (60, 40)
    kp[:, 11, :2], kp[:, 12, :2] = (42, 100), (58, 100)
    kp[:, 7, :2], kp[:, 8, :2] = (40, 70), (60, 70)
    kp[:, 9, :2], kp[:, 10, :2] = (40, 95), (60, 95)
    return VideoPoses(
        video_id="v",
        camera="c",
        n_frames=n,
        frame_idx=np.arange(n, dtype=np.int64),
        track_id=np.full(n, track, np.int64),
        bbox=np.tile(np.array([[30, 20, 70, 160]], np.float32), (n, 1)),
        keypoints=kp,
        frame_labels=None,
    )


def _move_right_hand(v: VideoPoses, frames: slice, x: float, y: float) -> None:
    v.keypoints[frames, 8, :2] = (x - 10, y - 10)
    v.keypoints[frames, 10, :2] = (x, y)


def test_hand_points_push_the_wrist_along_the_forearm() -> None:
    v = _person(1)
    p = hand_points(v.keypoints, 0.5)
    assert p[0].tolist() == [[40.0, 107.5], [60.0, 107.5]]
    v.keypoints[0, 9, 2] = 0.2
    assert np.allclose(hand_conf(v.keypoints)[0], [0.2, 0.9])


def test_contacts_are_runs_of_one_hand_in_the_shelf_zone() -> None:
    v = _person(60)
    _move_right_hand(v, slice(10, 20), 150, 50)
    _move_right_hand(v, slice(22, 25), 150, 50)  # a 2-frame gap is joined
    _move_right_hand(v, slice(40, 42), 150, 50)  # too short
    inside = hands_in_zone(hand_points(v.keypoints, 0.0), hand_conf(v.keypoints), SHELF, 0.3)
    assert contacts(v, inside, max_gap=2, min_frames=3) == [Contact(0, 1, 10, 24)]
    low = hands_in_zone(hand_points(v.keypoints, 0.0), hand_conf(v.keypoints) * 0, SHELF, 0.3)
    assert contacts(v, low, max_gap=2, min_frames=3) == []


def test_takes_compare_the_hold_before_and_after_the_contact() -> None:
    v = _person(60)
    hold = np.zeros((60, 2))
    hold[25:, 1] = 0.9
    hold[30, 1] = np.nan  # an unscored frame is skipped
    (t,) = takes(v, [Contact(0, 1, 10, 24)], hold, window=10)
    assert (t.frame, t.before, t.after) == (24, 0.0, 0.9)


def test_releases_name_the_place_where_the_hand_let_go() -> None:
    v = _person(100)
    hold = np.zeros((100, 2))
    hold[10:30, 1] = 0.9  # let go at frame 29 with the hand on the body (hips)
    hold[50:70, 1] = 0.9  # let go at frame 69 in the basket
    _move_right_hand(v, slice(66, 72), 25, 175)
    _move_right_hand(v, slice(26, 28), 25, 175)  # 2 of the last 3 frames near the body
    hold[80:100, 0] = 0.9  # still holding when the track ends: not a release
    points = hand_points(v.keypoints, 0.0)
    places = {"basket": hands_in_zone(points, hand_conf(v.keypoints), BASKET, 0.3)}
    body = near_body(points, v.keypoints, radius=1.0)
    out = releases(v, hold, 0.5, places, body, max_gap=2, min_frames=5, look=2)
    assert out == [Release(0, 1, 10, 29, "body"), Release(0, 1, 50, 69, "basket")]
    far = np.zeros_like(body)
    assert releases(v, hold, 0.5, {}, far, 2, 5, 2)[0].where == "unknown"


def test_near_body_scales_with_the_torso_and_needs_one() -> None:
    v = _person(2)
    p = np.array([[[50.0, 70.0], [50.0, 200.0]], [[50.0, 70.0], [50.0, 200.0]]])
    v.keypoints[1, [5, 6, 11, 12], :2] = 0  # no torso
    assert near_body(p, v.keypoints, 1.0).tolist() == [[True, False], [False, False]]


def test_shop_alerts_concealment_and_unpaid_items() -> None:
    took = Take(0, 1, 20, 0.1, 0.9)
    put_back = Take(0, 1, 40, 0.9, 0.1)
    other = Take(1, 0, 30, 0.1, 0.8)
    body = Release(0, 1, 21, 35, "body")
    exits = {0: [100], 1: [120]}
    got = shop_alerts([took, other], [body], exits, [], 0.5, grace=10)
    assert got == [
        ShopAlert(0, 35, "concealment", 1),
        ShopAlert(0, 100, "unpaid_exit", 1),
        ShopAlert(1, 120, "unpaid_exit", 1),
    ]
    # put back: nothing left to pay for
    assert shop_alerts([took, put_back], [], {0: [100]}, [], 0.5, 10) == []
    # paid within the grace, with and without an item count
    assert shop_alerts([took], [], {0: [100]}, [(105, 1)], 0.5, 10) == []
    assert shop_alerts([took], [], {0: [100]}, [(105, None)], 0.5, 10) == []
    # paid too late, or for fewer items than taken
    assert len(shop_alerts([took], [], {0: [100]}, [(111, 1)], 0.5, 10)) == 1
    two = [took, Take(0, 0, 25, 0.0, 0.9)]
    assert shop_alerts(two, [], {0: [100]}, [(90, 1)], 0.5, 10)[0].items == 2
    # a take after the exit belongs to the next visit
    assert shop_alerts([Take(0, 1, 150, 0.0, 0.9)], [], {0: [100]}, [], 0.5, 10) == []


def test_after_contact_keeps_only_releases_of_items_from_the_shelf() -> None:
    released = [
        Release(0, 1, 40, 60, "body"),  # contact ended at 35: within 10 frames before the hold
        Release(0, 0, 100, 120, "body"),  # no contact near it
        Release(1, 1, 40, 60, "body"),  # another person's contact does not count
        Release(0, 1, 130, 150, "body"),  # contact ending during the hold
    ]
    found = [Contact(0, 1, 20, 35), Contact(0, 0, 135, 140)]
    assert after_contact(released, found, 10) == [released[0], released[3]]
