"""PoseLift loader on synthetic data written in the real layout."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tcc.data.poselift import (
    PoseLiftFormatError,
    load_video,
    parse_video,
    read_pickle,
    sanitize,
)
from tests.fixtures.synthetic import SyntheticVideo, generate, poselift_id, write_poselift

CFG = {"keypoint_order": "yx_stored", "bbox_format": "xyxy", "nan_confidence": "interpolate"}


@pytest.fixture(scope="module")
def videos() -> list[SyntheticVideo]:
    return generate(seed=11, n_train=3, n_test=2)


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory, videos: list[SyntheticVideo]) -> Path:
    path = tmp_path_factory.mktemp("poselift")
    write_poselift(path, videos)
    return path


def _entry() -> list[Any]:
    return [[10, 20, 30, 60], np.ones((17, 3), np.float16)]


def test_loader_returns_one_row_per_person_per_frame(
    root: Path, videos: list[SyntheticVideo]
) -> None:
    for v in videos:
        loaded = load_video(root, v.split, poselift_id(v.name), CFG)
        assert len(loaded) == sum(len(idx) for idx, _ in v.people.values())
        assert loaded.camera == v.name[1]


def test_loader_swaps_stored_yx_into_contract_xy(root: Path, videos: list[SyntheticVideo]) -> None:
    v = videos[0]
    loaded = load_video(root, v.split, poselift_id(v.name), CFG)
    idx, seq = v.people[1]
    rows = loaded.track_id == 1
    np.testing.assert_array_equal(loaded.frame_idx[rows], idx)
    expected = seq.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(loaded.keypoints[rows], expected)


def test_loader_keeps_bbox_as_x1y1x2y2_containing_the_keypoints(
    root: Path, videos: list[SyntheticVideo]
) -> None:
    v = videos[1]
    loaded = load_video(root, v.split, poselift_id(v.name), CFG)
    x, y = loaded.keypoints[..., 0], loaded.keypoints[..., 1]
    assert (x >= loaded.bbox[:, :1] - 1).all() and (x <= loaded.bbox[:, 2:3] + 1).all()
    assert (y >= loaded.bbox[:, 1:2] - 1).all() and (y <= loaded.bbox[:, 3:4] + 1).all()


def test_frame_index_is_base_zero_and_labels_cover_the_video(
    root: Path, videos: list[SyntheticVideo]
) -> None:
    test_video = next(v for v in videos if v.split == "test")
    loaded = load_video(root, "test", poselift_id(test_video.name), CFG)
    assert loaded.frame_labels is not None
    np.testing.assert_array_equal(loaded.frame_labels, test_video.labels)
    assert loaded.n_frames == len(test_video.labels)
    records = list(loaded.records())
    assert records[0].label == int(test_video.labels[records[0].frame_idx])


def test_train_videos_are_unlabelled(root: Path, videos: list[SyntheticVideo]) -> None:
    v = next(v for v in videos if v.split == "train")
    loaded = load_video(root, "train", poselift_id(v.name), CFG)
    assert loaded.frame_labels is None
    assert next(loaded.records()).label is None


def test_frames_not_starting_at_zero_fail() -> None:
    with pytest.raises(PoseLiftFormatError, match="0..n-1"):
        parse_video("1_1", {1: {}, 2: {}}, None, "yx_stored", "xyxy", "one")


def test_entry_that_is_not_bbox_and_keypoints_fails() -> None:
    with pytest.raises(PoseLiftFormatError, match=r"\[bbox, kp\]"):
        parse_video("1_1", {0: {3: {"keypoints": []}}}, None, "yx_stored", "xyxy", "one")


def test_wrong_keypoint_shape_fails() -> None:
    raw = {0: {3: [[1, 2, 3, 4], np.ones((18, 3), np.float16)]}}
    with pytest.raises(PoseLiftFormatError, match="keypoints"):
        parse_video("1_1", raw, None, "yx_stored", "xyxy", "one")


def test_labels_shorter_than_pose_frames_fail() -> None:
    raw = {0: {}, 1: {3: _entry()}}
    with pytest.raises(PoseLiftFormatError, match="labels"):
        parse_video("1_1", raw, np.zeros(1, np.int64), "yx_stored", "xyxy", "one")


def test_bad_video_id_fails() -> None:
    with pytest.raises(PoseLiftFormatError, match="camera"):
        parse_video("C1_1", {0: {3: _entry()}}, None, "yx_stored", "xyxy", "one")


def test_pickle_requesting_other_globals_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "evil.pkl"
    with path.open("wb") as f:
        pickle.dump({0: {1: [Path("x"), None]}}, f)
    with pytest.raises(PoseLiftFormatError, match="forbidden global"):
        read_pickle(path)


def test_sanitize_replaces_nan_confidence_and_zeroes_nan_coordinates() -> None:
    kp = np.ones((2, 17, 3), np.float32)
    kp[0, 3, 2] = np.nan
    kp[1, 4, 0] = np.nan
    out, n_conf, n_xy = sanitize(kp, conf_nan=0.5)
    assert (n_conf, n_xy) == (1, 1)
    assert out[0, 3, 2] == 0.5
    np.testing.assert_array_equal(out[1, 4], [0.0, 0.0, 0.0])
    assert np.isfinite(out).all()


def test_sanitize_with_none_keeps_raw_values() -> None:
    kp = np.ones((1, 17, 3), np.float32)
    kp[0, 0, 2] = np.nan
    out, n_conf, _ = sanitize(kp, conf_nan=None)
    assert n_conf == 1 and np.isnan(out[0, 0, 2])


def _nan_track() -> dict[int, dict[int, list[Any]]]:
    """Track 3 over frames 0..3; frames 1 and 2 have NaN confidence (interpolated pose)."""
    raw: dict[int, dict[int, list[Any]]] = {}
    for q, conf in enumerate([0.2, np.nan, np.nan, 0.8]):
        kp = np.full((17, 3), conf, np.float16)
        kp[:, 0], kp[:, 1] = 100 + q, 50 + q  # stored (y, x)
        raw[q] = {3: [[40, 90, 60, 110], kp]}
    return raw


def test_nan_confidence_interpolate_uses_neighbouring_real_poses() -> None:
    v = parse_video("1_1", _nan_track(), None, "yx_stored", "xyxy", "interpolate")
    expected = np.array([0.2, 0.4, 0.6, 0.8], np.float16).astype(np.float32)
    np.testing.assert_allclose(v.keypoints[:, 0, 2], expected, atol=1e-3)


def test_nan_confidence_one_and_zero() -> None:
    one = parse_video("1_1", _nan_track(), None, "yx_stored", "xyxy", "one")
    zero = parse_video("1_1", _nan_track(), None, "yx_stored", "xyxy", "zero")
    assert one.keypoints[1:3, :, 2].tolist() == [[1.0] * 17] * 2
    assert zero.keypoints[1:3, :, 2].tolist() == [[0.0] * 17] * 2


def test_nan_confidence_keep_leaves_nan() -> None:
    v = parse_video("1_1", _nan_track(), None, "yx_stored", "xyxy", "keep")
    assert np.isnan(v.keypoints[1:3, :, 2]).all()


def test_nan_confidence_at_track_edge_fails_interpolation() -> None:
    raw = _nan_track()
    raw[3][3][1][:, 2] = np.nan
    with pytest.raises(PoseLiftFormatError, match="both sides"):
        parse_video("1_1", raw, None, "yx_stored", "xyxy", "interpolate")


def test_keypoint_order_as_is_keeps_stored_columns() -> None:
    v = parse_video("1_1", _nan_track(), None, "as_is", "xyxy", "one")
    assert v.keypoints[0, 0, :2].tolist() == [100.0, 50.0]
    w = parse_video("1_1", _nan_track(), None, "yx_stored", "xyxy", "one")
    assert w.keypoints[0, 0, :2].tolist() == [50.0, 100.0]
