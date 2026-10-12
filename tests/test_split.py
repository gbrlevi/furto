"""The official split is read from the dataset and nothing else (rule 1)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from tcc.data.split import MissingOfficialSplitError, official_split
from tests.fixtures.synthetic import generate, poselift_id, write_poselift


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_poselift(tmp_path, generate(seed=4, n_train=3, n_test=2))
    return tmp_path


def test_official_split_comes_from_train_and_test_folders(root: Path) -> None:
    videos = generate(seed=4, n_train=3, n_test=2)
    split = official_split(root)
    assert split["train"] == sorted(poselift_id(v.name) for v in videos if v.split == "train")
    assert split["test"] == sorted(poselift_id(v.name) for v in videos if v.split == "test")


def test_missing_split_folder_fails(root: Path) -> None:
    shutil.rmtree(root / "Pickle_files" / "Test")
    with pytest.raises(MissingOfficialSplitError, match="not found"):
        official_split(root)


def test_flat_folder_without_split_fails(tmp_path: Path) -> None:
    (tmp_path / "Pickle_files").mkdir()
    with pytest.raises(MissingOfficialSplitError):
        official_split(tmp_path)


def test_video_in_both_partitions_fails(root: Path) -> None:
    test_id = official_split(root)["test"][0]
    shutil.copy(root / "Pickle_files/Test" / f"{test_id}.pkl", root / "Pickle_files/Train")
    with pytest.raises(MissingOfficialSplitError, match="both"):
        official_split(root)


def test_test_video_without_labels_fails(root: Path) -> None:
    test_id = official_split(root)["test"][0]
    (root / "Pickle_files/GT" / f"{test_id}.npy").unlink()
    with pytest.raises(MissingOfficialSplitError, match="without labels"):
        official_split(root)


def test_labels_for_a_train_video_fail(root: Path) -> None:
    train_id = official_split(root)["train"][0]
    np.save(root / "Pickle_files/GT" / f"{train_id}.npy", np.zeros(5, np.int64))
    with pytest.raises(MissingOfficialSplitError, match="outside the test split"):
        official_split(root)
