import numpy as np
import pytest

from tcc.data.schema import VideoPoses


def _video(**overrides: object) -> VideoPoses:
    fields: dict[str, object] = {
        "video_id": "1_1",
        "camera": "1",
        "n_frames": 3,
        "frame_idx": np.array([0, 1, 1], np.int64),
        "track_id": np.array([5, 5, 6], np.int64),
        "bbox": np.array([[0, 0, 10, 20]] * 3, np.float32),
        "keypoints": np.zeros((3, 17, 3), np.float32),
        "frame_labels": np.array([0, 1, 0], np.int64),
    }
    fields.update(overrides)
    return VideoPoses(**fields)  # type: ignore[arg-type]


def test_valid_video_yields_one_record_per_row() -> None:
    records = list(_video().records())
    assert [(r.frame_idx, r.track_id, r.label) for r in records] == [
        (0, 5, 0),
        (1, 5, 1),
        (1, 6, 1),
    ]
    assert records[0].bbox == (0.0, 0.0, 10.0, 20.0)
    assert records[0].keypoints.shape == (17, 3)


def test_frame_out_of_range_fails() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _video(frame_idx=np.array([0, 1, 3], np.int64))


def test_duplicate_person_in_frame_fails() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _video(track_id=np.array([5, 6, 6], np.int64))


def test_inverted_bbox_fails() -> None:
    with pytest.raises(ValueError, match="x2 < x1"):
        _video(bbox=np.array([[10, 0, 0, 20]] * 3, np.float32))


def test_labels_must_cover_every_frame() -> None:
    with pytest.raises(ValueError, match="one value per frame"):
        _video(frame_labels=np.array([0, 1], np.int64))


def test_wrong_keypoint_shape_fails() -> None:
    with pytest.raises(ValueError, match="keypoints"):
        _video(keypoints=np.zeros((3, 18, 3), np.float32))
