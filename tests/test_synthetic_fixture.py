import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.synthetic import generate, write_pkl

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "legacy" / "poc_luiz" / "ferramentas" / "gerar_dados_falsos.py"


def test_same_seed_gives_identical_videos() -> None:
    a, b = generate(seed=5, n_train=3, n_test=2), generate(seed=5, n_train=3, n_test=2)
    for va, vb in zip(a, b, strict=True):
        assert va.name == vb.name and va.split == vb.split
        np.testing.assert_array_equal(va.labels, vb.labels)
        assert va.people.keys() == vb.people.keys()
        for pid in va.people:
            np.testing.assert_array_equal(va.people[pid][0], vb.people[pid][0])
            np.testing.assert_array_equal(va.people[pid][1], vb.people[pid][1])


def test_different_seeds_give_different_videos() -> None:
    a, b = generate(seed=1, n_train=1, n_test=0), generate(seed=2, n_train=1, n_test=0)
    assert not np.array_equal(a[0].people[1][1], b[0].people[1][1])


def test_train_videos_have_no_theft_and_test_videos_have_theft() -> None:
    videos = generate(seed=1, n_train=4, n_test=3)
    assert [v.split for v in videos] == ["train"] * 4 + ["test"] * 3
    assert all(v.labels.sum() == 0 for v in videos if v.split == "train")
    assert all(v.labels.sum() > 0 for v in videos if v.split == "test")


def test_people_shapes_and_frame_ranges() -> None:
    for v in generate(seed=3, n_train=2, n_test=2):
        for idx, seq in v.people.values():
            assert seq.shape == (len(idx), 17, 3)
            assert idx.min() >= 0 and idx.max() < len(v.labels)


@pytest.mark.skipif(not ORIGINAL.exists(), reason="legacy copy removed after the port")
def test_port_reproduces_original_script_for_the_same_seed(tmp_path: Path) -> None:
    """Characterization: run the original as a subprocess (no import) and compare."""
    original_dir, port_dir = tmp_path / "original", tmp_path / "port"
    cmd = [sys.executable, str(ORIGINAL), "--saida", str(original_dir)]
    cmd += ["--treino", "3", "--teste", "2", "--semente", "7"]
    subprocess.run(cmd, check=True, capture_output=True)
    write_pkl(port_dir, generate(seed=7, n_train=3, n_test=2))

    names = sorted(p.name for p in original_dir.iterdir())
    assert names == sorted(p.name for p in port_dir.iterdir())
    for name in names:
        if name.endswith(".npy"):
            np.testing.assert_array_equal(np.load(original_dir / name), np.load(port_dir / name))
            continue
        with open(original_dir / name, "rb") as f:
            original = pickle.load(f)
        with open(port_dir / name, "rb") as f:
            port = pickle.load(f)
        assert original.keys() == port.keys()
        for q in original:
            for po, pp in zip(original[q], port[q], strict=True):
                assert po["person_id"] == pp["person_id"]
                assert po["bbox"] == pp["bbox"]
                np.testing.assert_array_equal(po["keypoints"], pp["keypoints"])
